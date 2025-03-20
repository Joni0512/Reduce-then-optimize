import logging
from datetime import datetime
from datetime import timedelta
import time
from handlers.request_handler import RequestHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler
from handlers.output_handler import OutputHandler
from handlers.payload_parser import PayloadParser
import argparse
import pickle 

SOLVER_TIMEOUT = 120
PENALTY = 1000000 # penalty for not serving a trip
SHAREABLE_COST_FACTOR = 1
MAX_CARDINALITY = 8
MAX_THREAD_CNT = 500
MAX_BATCH_SIZE = 100
BUS_CAPACITY = 50
MAX_WAITING = 1800
MAX_DETOUR = 1800
RH_FACTOR = 0
REBALANCING = False
RTV_TIMEOUT = 3000

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Simulator arguments')
    parser.add_argument('--max_cardinality', type=int,
                    help='maximum trips to be shared')
    parser.add_argument('--rh_factor', type=int,
                    help='RH FACTOR')
    parser.add_argument('--interval', type=int,
                    help='Batch interval')
    parser.add_argument('--out_put_dir',
                    help='output directory')
    parser.add_argument('--server_url',
                    help='Server URL')
    parser.add_argument('--input_file',
                    help='Request file')
    args = parser.parse_args()
    OUTPUT_DIR = args.out_put_dir
    MAX_CARDINALITY = args.max_cardinality
    RH_FACTOR = args.rh_factor
    BATCH_INTERVAL = timedelta(0,seconds=args.interval)

    logging.basicConfig(filename=OUTPUT_DIR+'main.log', level=logging.INFO)
    logging.info('Starting the simulator with: Batch Interval {0}, RH FACTOR {1}'.format(BATCH_INTERVAL, RH_FACTOR))
    iteration = 0
    NetworkHandler.init(True,args.server_url)

    with open(args.input_file, 'rb') as f:
        payload = pickle.load(f)
    start_of_the_day = datetime.strptime(payload['date'], '%Y-%m-%d')
    dwell_pickup = 180
    dwell_alight = 60

    payload_object = PayloadParser.get_payload_object(payload,False)
    start_of_the_day = payload_object.start_of_the_day
    request_handler = RequestHandler(payload_object.requests, start_of_the_day, dwell_pickup, dwell_alight)
    starting_time = request_handler.earliest_start_time()
    latest_time = request_handler.latest_start_time()
    vehicle_handler = VehicleHandler(payload_object.depot, payload_object.driver_runs,OUTPUT_DIR,start_of_the_day)
    output_handler = OutputHandler(OUTPUT_DIR)

    active_requests = {}
    boarded_requests = {}

    starting_time = vehicle_handler.earliest_start_time-BATCH_INTERVAL
    while starting_time <= latest_time or (len(active_requests) + len(boarded_requests) > 0):
        iteration_exe_start_time = time.time()
        end_time = starting_time + BATCH_INTERVAL
        batch = []
        current_batch,end_time = request_handler.get_batch(end_time,MAX_BATCH_SIZE)
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
        for req_id in picked_requests:
            boarded_requests[req_id] = active_requests[req_id]
            active_requests.pop(req_id)
        for req_id in completed_requests:
            boarded_requests.pop(req_id)
        output_handler.record_vehicles(vehicle_handler.get_vehicle_locations(),end_time)
        output_handler.record_completed_stops(completed_stops)
        if len(batch) + len(active_requests) > 0 :
            for req_id in active_requests:
                batch.append(active_requests[req_id])
            trip_handler = TripHandler(end_time,vehicle_handler.vehicles,batch, active_requests, iteration, SOLVER_TIMEOUT,PENALTY,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR,REBALANCING, RTV_TIMEOUT)
            output_handler.record_output(end_time,batch,trip_handler,time.time()-iteration_exe_start_time)

            active_requests = {}
            for request_id in trip_handler.request_assignment:
                for request in batch:
                    if request.id == request_id:
                        active_requests[request_id] = request
                        break

            for vehicle_id in trip_handler.vehicle_assignment:
                vehicle = vehicle_handler.vehicles[vehicle_id]
                trips = trip_handler.vehicle_assignment[vehicle_id]
                VehicleHandler.add_new_trips(end_time, vehicle, trips, add=True)

            rebalancing_trip_info = []
            for vehicle_id in trip_handler.rebalancing_assignment:
                vehicle = vehicle_handler.vehicles[vehicle_id]
                destination = trip_handler.rebalancing_assignment[vehicle_id]
                VehicleHandler.add_rebalancing_trip(vehicle, destination,end_time)
                rebalancing_trip_info.append([vehicle_id,vehicle.last_node,destination,vehicle.time_at_last])
            output_handler.record_rebalancing_trips(rebalancing_trip_info,end_time)
        starting_time = end_time
        iteration+=1

        # create updated driver runs
        updated_driver_runs = []
        for driver_run in payload["driver_runs"]:
            new_driver_run = vehicle_handler.get_state(driver_run,start_of_the_day)
            updated_driver_runs.append(new_driver_run)
        
        payload["driver_runs"] = updated_driver_runs

        formatted_end_time = end_time.strftime('%H%M%S')
        with open(OUTPUT_DIR+'manifests/state_{0}.pkl'.format(formatted_end_time), 'wb') as file:
            pickle.dump(payload, file)
    request_handler.requests.to_csv(output_handler.output_directory+"requests.csv",index=False)
