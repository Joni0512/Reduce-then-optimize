from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.handlers.output_handler import OutputHandler
from rtv_solver.handlers.payload_parser import PayloadParser
import argparse
import pickle 
import os
import sys
import multiprocessing
import logging
from datetime import datetime, timedelta
import time

SOLVER_TIMEOUT = 120
PENALTY = 1000000 # penalty for not serving a trip
SHAREABLE_COST_FACTOR = 1
MAX_THREAD_CNT = 1 # JW: 8 threads as I can have 8 
MAX_BATCH_SIZE = 400 # JW: reduced for now as it reduces the complexity 
BUS_CAPACITY = 0 # JW: what is this?
MAX_WAITING = 7200 # presumably in seconds, tested with longer waiting times to ease matching
MAX_DETOUR = 7200 # presumably in secondsç
RH_FACTOR = 0 # factor for additional time step to extend optimization horizon
REBALANCING = False
RTV_TIMEOUT = 3000

DEBUG_BOOL = True

# TODO
# shared trips does not seem to work
# never stops because last boarded request is never dropped off or at least our code does not recognize it

if __name__=="__main__":
    """
    Learnings JW:
    1. The current code only works with specific datasets as the structure of the underlying payloads has been changed between the most recent branch and the master one. Input dataset "localDB_payload_oct.pkl" does not work and presumably "oct_payload_4_00.pkl" and "oct_payload_7_30.pkl" will not work either as they are older.
    """
    # for macOS set start method to fork to avoid issues with multiprocessing and requests
    if sys.platform == "darwin":
            multiprocessing.set_start_method("fork")
    # parse arguments
    # increase this level
    parser = argparse.ArgumentParser(description='Simulator arguments')
    parser.add_argument('--max_cardinality', type=int, default=4,help='maximum trips to be shared')
    parser.add_argument('--rh_factor', type=int,default=0,help='RH FACTOR')
    parser.add_argument('--interval', type=int,default=3600,help='Batch interval in seconds') # test with 3600 so it runs faster under the simple setting
    parser.add_argument('--out_put_dir', type=str,default="output_format/debug/",help='output directory')
    parser.add_argument('--server_url', type=str,default="http://127.0.0.1:5001/",help='Server URL')
    parser.add_argument('--input_file', type=str,default="rtv-solver/inputs/wilson_nc_initial.pkl",help='Request file')
    args = parser.parse_args()
    print(args)

    OUTPUT_DIR = args.out_put_dir
    MAX_CARDINALITY = args.max_cardinality
    RH_FACTOR = args.rh_factor
    BATCH_INTERVAL = timedelta(days=0,seconds=args.interval)
    MAX_THREAD_CNT = min(MAX_THREAD_CNT, os.cpu_count())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(filename=OUTPUT_DIR+'main.log', level=logging.INFO)
    logging.info('Starting the simulator with: Batch Interval {0}, RH FACTOR {1}'.format(BATCH_INTERVAL, RH_FACTOR))
    iteration = 1

    with open(args.input_file, 'rb') as f:
        payload = pickle.load(f)
    start_of_the_day = datetime.strptime(payload['date'], '%Y-%m-%d')
    dwell_pickup = 180
    dwell_alight = 60

    # for debugging only consider a single vehicle
    if DEBUG_BOOL:
            payload[PayloadParser.DRIVERS] = payload[PayloadParser.DRIVERS][:1]

    payload_object = PayloadParser.get_payload_object(payload,False)
    request_handler = RequestHandler(payload_object.requests, dwell_pickup, dwell_alight)      
    vehicle_handler = VehicleHandler(payload_object.depot, payload_object.driver_runs, OUTPUT_DIR)
    output_handler = OutputHandler(OUTPUT_DIR)

    starting_time = request_handler.earliest_start_time()
    latest_time = request_handler.latest_start_time()   

    active_requests = {}
    boarded_requests = {}

    starting_time = vehicle_handler.earliest_start_time - BATCH_INTERVAL.total_seconds()
    while starting_time <= latest_time or (len(active_requests) + len(boarded_requests) > 0):
        if starting_time >= 67435.0:
            print("debug")
        print("Iteration ", iteration, "starting time:", starting_time, "latest time:", latest_time, "active requests:", len(active_requests), "boarded requests:", len(boarded_requests))
        iteration_exe_start_time = time.time()
        end_time = starting_time + BATCH_INTERVAL.total_seconds()
        batch = []
        current_batch, end_time = request_handler.get_batch(end_time,MAX_BATCH_SIZE)
        for requests in current_batch:
            if requests.id not in active_requests:
                batch.append(requests)
        future_trips = request_handler.get_lookahead_trips(end_time,RH_FACTOR,BATCH_INTERVAL)
        for requests in future_trips:
            if requests.id not in active_requests:
                batch.append(requests)
        completed_stops, picked_requests, completed_requests = vehicle_handler.simulate_vehicles(end_time)
        for stop in completed_stops:
            for driver_run in payload["driver_runs"]:
                if driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_RUN_ID] == stop.vehicle_id:
                    driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_LOC_SERV] += 1
                    break
        
        # FIXME: why are there some requests double in the list of picked_requests? seems to be a bug in simulate_vehicles?
        for req_id in picked_requests:
            boarded_requests[req_id] = active_requests[req_id]
            active_requests.pop(req_id)
        for req_id in completed_requests:
            boarded_requests.pop(req_id)
        
        # JW: does that need to be here or can we bundle Output processing with below after TripHandler
        output_handler.record_vehicles(vehicle_handler.get_vehicle_locations(), end_time)
        output_handler.record_completed_stops(completed_stops)
        
        # FIXME test, why do the trips not have to be calculated for boarded requests
        # easier to debug to see lengths in-line
        batch_len, active_len, boarded_len = len(batch), len(active_requests), len(boarded_requests)
        if batch_len + active_len + boarded_len > 0 :
            for req_id in active_requests:
                batch.append(active_requests[req_id])
            
            # FIXME: why is the assignment not deterministic?
            trip_handler = TripHandler( 
                vehicle_handler.vehicles,
                batch, 
                active_requests, 
                iteration, 
                SOLVER_TIMEOUT,
                PENALTY,
                MAX_CARDINALITY,
                MAX_THREAD_CNT,
                SHAREABLE_COST_FACTOR,
                REBALANCING,
                RTV_TIMEOUT)

            perf_duration = time.time()-iteration_exe_start_time
            output_handler.record_output(end_time, batch, trip_handler, perf_duration)

            # TODO update to dictionary-based version with indexed batch; change only after entire code runs through and performance improvement is valid
            # batch_by_id = {request.id: request for request in batch} # Build batch lookup table once
            # active_requests = {request_id: batch_by_id[request_id] for request_id in trip_handler.request_assignment if request_id in batch_by_id} # Select only active requests
            # active_requests = {} # this overwrites the real counter and should not be here
            batch_by_id = {request.id: request for request in batch} # Build batch lookup table once
            active_requests = {request_id: batch_by_id[request_id] for request_id in trip_handler.request_assignment if request_id in batch_by_id} # 
            # for request_id in trip_handler.request_assignment:
              #   for request in batch:
                #     if request.id == request_id:
                  #      active_requests[request_id] = request
                   #     break

            for vehicle_id in trip_handler.vehicle_assignment:
                vehicle = vehicle_handler.vehicles[vehicle_id]
                trips, trip_sequence = trip_handler.vehicle_assignment[vehicle_id]
                VehicleHandler.add_new_trips(vehicle, trips, trip_sequence, add=True)

            rebalancing_trip_info = []
            for vehicle_id in trip_handler.rebalancing_assignment:
                vehicle = vehicle_handler.vehicles[vehicle_id]
                destination = trip_handler.rebalancing_assignment[vehicle_id]
                VehicleHandler.add_rebalancing_trip(vehicle, destination,end_time)
                rebalancing_trip_info.append([vehicle_id,vehicle.last_node,destination,vehicle.time_at_last])
            output_handler.record_rebalancing_trips(rebalancing_trip_info,end_time)
        starting_time = end_time
        iteration += 1
        # print("Iteration", iteration)

        # update driver runs
        # print("Completed vehicles - main:", completed_vehicles)
        updated_driver_runs = []
        for driver_run in payload["driver_runs"]:
            new_driver_run = vehicle_handler.get_state(driver_run)
            if new_driver_run is not None:
                updated_driver_runs.append(new_driver_run)
        
        payload["driver_runs"] = updated_driver_runs

        formatted_end_time = int(end_time) # change >> .strftime('%H%M%S'); JW: int() simplify for now
        with open(OUTPUT_DIR+'manifests/state_{0}.pkl'.format(formatted_end_time), 'wb') as file:
            pickle.dump(payload, file)
    request_handler.requests.to_csv(output_handler.output_directory+"requests.csv",index=False)
