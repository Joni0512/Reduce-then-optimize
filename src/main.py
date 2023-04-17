import logging
from datetime import timedelta
import time
from structure.assignment import AssignmentWithBus
from handlers.request_handler import RequestHandler
from handlers.bus_handler import BusHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler
from handlers.output_handler import OutputHandler
import argparse
import multiprocessing
import numpy as np


BATCH_INTERVAL = timedelta(0,seconds=30)
WALK_DISTANCE_CUT_OFF = 500 #maximum walkable distance to a bus stop (meters)
BUS_DISTANCE_CUT_OFF = 2000 #consider only the lines that are within distance (meters)
IPM_SOLVER_TIMEOUT = 1200
PENALTY = 1000000 # penalty for not serving a trip
MINIMUM_TRIP_DURATION = 900 # 15 minutes
ADDITIONAL_TRIP_TIME_FACTOR = 1 #we allow trips to have 2 x shortest path travel distance
SHAREABLE_COST_FACTOR = 1
MAX_CARDINALITY = 2
MAX_THREAD_CNT = 10
MAX_BATCH_SIZE = 100
BUS_CAPACITY = 50
USE_REAL_DISTANCE = False # if False, assume distance is propotional to the travel time

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Simulator arguments')
    parser.add_argument('--max_number_of_vehicles', type=int,
                    help='maximum number of MoD vehicles')
    parser.add_argument('--max_capacity', type=int,
                    help='maximum capacity of a MoD vehicle')
    parser.add_argument('--max_cardinality', type=int,
                    help='maximum trips to be shared')
    parser.add_argument('--allow_bus_transfer', action=argparse.BooleanOptionalAction,
                    help='allow bus transfers')
    parser.add_argument('--allow_bus', action=argparse.BooleanOptionalAction,
                    help='allow busses')
    parser.add_argument('--out_put_dir',
                    help='output directory')
    parser.add_argument('--data_dir',
                    help='data directory')
    args = parser.parse_args()
    BASE_DATA_DIR = args.data_dir
    OUTPUT_DIR = args.out_put_dir
    MAX_CARDINALITY = args.max_cardinality

    logging.basicConfig(filename=OUTPUT_DIR+'main.log', level=logging.INFO)
    logging.info('Starting the simulator with: max_number_of_vehicles {0}, max_capacity {1}'.format(args.max_number_of_vehicles, args.max_capacity))
    iteration = 0
    NetworkHandler.init(BASE_DATA_DIR+"map/10km/",USE_REAL_DISTANCE)

    request_handler = RequestHandler(BASE_DATA_DIR+"requests/requests_10km.csv",MINIMUM_TRIP_DURATION,ADDITIONAL_TRIP_TIME_FACTOR)
    starting_time = request_handler.earliest_start_time()
    latest_time = request_handler.latest_start_time()
    start_of_the_day = starting_time.replace(hour=0, minute=0, second=0, microsecond=0)
    bus_handler = BusHandler(BASE_DATA_DIR+"bus/MBTA_GTFS/", start_of_the_day, BUS_CAPACITY,False)
    vehicle_handler = VehicleHandler(BASE_DATA_DIR+"vehicles/vehicles_10km.csv",OUTPUT_DIR,start_of_the_day,args.max_number_of_vehicles, args.max_capacity)

    # OUTPUT_DIR = "../output/"
    output_handler = OutputHandler(OUTPUT_DIR)

    def get_bus_trips(request_no):
        return bus_handler.generate_bus_trips(batch[request_no], args.allow_bus, args.allow_bus_transfer,WALK_DISTANCE_CUT_OFF)

    while starting_time <= latest_time:
        iteration_exe_start_time = time.time()
        end_time = starting_time + BATCH_INTERVAL
        batch,end_time = request_handler.get_batch(end_time,MAX_BATCH_SIZE)
        completed_stops = vehicle_handler.simulate_vehicles(end_time)
        output_handler.record_vehicles(vehicle_handler.get_vehicle_locations(end_time),end_time)
        output_handler.record_completed_stops(completed_stops)
        if len(batch) > 0:
            request_bus_combinations = {}
            pool = multiprocessing.Pool(processes=MAX_THREAD_CNT)
            request_numbers = np.arange(len(batch))
            combinations = pool.map(get_bus_trips,request_numbers)
            request_no = 0
            for request in batch:
                request_bus_combinations[request.id] = combinations[request_no]
                request_no+=1
            combinations = None

            trip_handler = TripHandler(end_time,vehicle_handler.vehicles,batch,request_bus_combinations, WALK_DISTANCE_CUT_OFF,IPM_SOLVER_TIMEOUT,PENALTY,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR)
            output_handler.record_output(end_time,batch,trip_handler,time.time()-iteration_exe_start_time)

            for request_id in trip_handler.request_assignment:
                assignment  = trip_handler.request_assignment[request_id]
                if isinstance(assignment,AssignmentWithBus):
                    bus_handler.add_passenger_to_bus_run(assignment.bus_trip)

            # if trip_handler.unassigned_trip_count > 0:
            # if end_time > ending_time:
            #     vehicle_handler.save_snapshot()
            #     break

            for vehicle_id in trip_handler.vehicle_assignment:
                vehicle = vehicle_handler.vehicles[vehicle_id]
                trips = trip_handler.vehicle_assignment[vehicle_id]
                VehicleHandler.add_new_trips(end_time, vehicle, trips, add=True)

            # for vehicle_id in trip_handler.rebalancing_assignment:
            #     vehicle = vehicle_handler.vehicles[vehicle_id]
            #     destination = trip_handler.rebalancing_assignment[vehicle_id]
            #     VehicleHandler.add_rebalancing_trip(vehicle, destination,end_time)
        starting_time = end_time
    
    output_handler.record_bus_usage(bus_handler.busslines)
