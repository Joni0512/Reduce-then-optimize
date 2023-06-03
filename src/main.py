import logging
from datetime import timedelta
import time
from handlers.request_handler import RequestHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler
from handlers.output_handler import OutputHandler
import argparse

IPM_SOLVER_TIMEOUT = 1200
PENALTY = 1000000 # penalty for not serving a trip
SHAREABLE_COST_FACTOR = 1
MAX_CARDINALITY = 2
MAX_THREAD_CNT = 500
MAX_BATCH_SIZE = 100
BUS_CAPACITY = 50
MAX_WAITING = 1800
MAX_DETOUR = 1800
RH_FACTOR = 0

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Simulator arguments')
    parser.add_argument('--max_number_of_vehicles', type=int,
                    help='maximum number of MoD vehicles')
    parser.add_argument('--max_capacity', type=int,
                    help='maximum capacity of a MoD vehicle')
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
    parser.add_argument('--request_file',
                    help='Request file')
    parser.add_argument('--vehicle_file',
                    help='Vehicle file')
    args = parser.parse_args()
    OUTPUT_DIR = args.out_put_dir
    MAX_CARDINALITY = args.max_cardinality
    RH_FACTOR = args.rh_factor
    BATCH_INTERVAL = timedelta(0,seconds=args.interval)

    logging.basicConfig(filename=OUTPUT_DIR+'main.log', level=logging.INFO)
    logging.info('Starting the simulator with: max_number_of_vehicles {0}, max_capacity {1}'.format(args.max_number_of_vehicles, args.max_capacity))
    logging.info('Batch Interval {0}, RH FACTOR {1}'.format(BATCH_INTERVAL, RH_FACTOR))
    logging.info('Max waiting time {0}, max detour time {1}'.format(MAX_WAITING, MAX_DETOUR))
    iteration = 0
    NetworkHandler.init(args.server_url)

    request_handler = RequestHandler(args.request_file,MAX_DETOUR,MAX_WAITING)
    starting_time = request_handler.earliest_start_time()
    latest_time = request_handler.latest_start_time()
    start_of_the_day = starting_time.replace(hour=0, minute=0, second=0, microsecond=0)
    vehicle_handler = VehicleHandler(args.vehicle_file,OUTPUT_DIR,start_of_the_day,args.max_number_of_vehicles, args.max_capacity)
    output_handler = OutputHandler(OUTPUT_DIR)

    active_requests = {}
    boarded_requests = {}

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
        for req_id in completed_requests:
            boarded_requests.pop(req_id)
        for req_id in picked_requests:
            boarded_requests[req_id] = active_requests[req_id]
            active_requests.pop(req_id)
        output_handler.record_vehicles(vehicle_handler.get_vehicle_locations(),end_time)
        output_handler.record_completed_stops(completed_stops)
        if len(batch) + len(active_requests) > 0 :
            for req_id in active_requests:
                batch.append(active_requests[req_id])
            trip_handler = TripHandler(end_time,vehicle_handler.vehicles,batch, active_requests, iteration, IPM_SOLVER_TIMEOUT,PENALTY,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR)
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
    