import logging
from datetime import timedelta
import time
from handlers.request_handler import RequestHandler
from handlers.bus_handler import BusHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler
from handlers.output_handler import OutputHandler
import argparse
import multiprocessing
import numpy as np

# import datetime

# ending_time = datetime.datetime(2015,1,1,0,12,0)

BASE_DATA_DIR = "../data/"
BATCH_INTERVAL = timedelta(0,seconds=30)
BUS_DWELL = 25 # second
AVERAGE_EDGE_SPEED = 10.785 #mps
WALK_DISTANCE_CUT_OFF = 500 #meters
BUS_DISTANCE_CUT_OFF = 1000 #meters
IPM_SOLVER_TIMEOUT = 1200
PENALTY = 1000000
MAX_WAIT_TIME = 300 # 5 minutes
ADDITIONAL_TRIP_TIME_FACTOR = 3 #we allow trips to have 3 x shortest path travel distance
SHAREABLE_COST_FACTOR = 1
MAX_CARDINALITY = 2
MAX_THREAD_CNT = 10
MAX_BATCH_SIZE = 100

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
    args = parser.parse_args()
    OUTPUT_DIR = args.out_put_dir
    MAX_CARDINALITY = args.max_cardinality

    logging.basicConfig(filename=OUTPUT_DIR+'main.log', level=logging.DEBUG)
    logging.info('Starting the simulator with: max_number_of_vehicles {0}, max_capacity {1}'.format(args.max_number_of_vehicles, args.max_capacity))
    iteration = 0
    NetworkHandler.init(BASE_DATA_DIR+"map/")

    request_handler = RequestHandler(BASE_DATA_DIR+"requests/requests.csv",MAX_WAIT_TIME,ADDITIONAL_TRIP_TIME_FACTOR)
    starting_time = request_handler.earliest_start_time()
    latest_time = request_handler.latest_start_time()
    start_of_the_day = starting_time.replace(hour=0, minute=0, second=0, microsecond=0)
    bus_handler = BusHandler(BASE_DATA_DIR+"bus/", start_of_the_day, BUS_DWELL, BUS_DISTANCE_CUT_OFF)
    vehicle_handler = VehicleHandler(BASE_DATA_DIR+"vehicles/vehicles.csv",OUTPUT_DIR,start_of_the_day,args.max_number_of_vehicles, args.max_capacity)

    # OUTPUT_DIR = "../output/"
    output_handler = OutputHandler(OUTPUT_DIR)

    def get_bus_trips(request_no):
        return bus_handler.generate_bus_trips(batch[request_no], args.allow_bus, args.allow_bus_transfer)

    while starting_time <= latest_time:
        iteration_exe_start_time = time.time()

        end_time = starting_time + BATCH_INTERVAL
        batch,end_time = request_handler.get_batch(end_time,MAX_BATCH_SIZE)
        vehicle_handler.simulate_vehicles(end_time)

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
        # if trip_handler.unassigned_trip_count > 0:
        # if end_time > ending_time:
        #     vehicle_handler.save_snapshot()
        #     break
        for vehicle_id in trip_handler.vehicle_assignment:
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips = trip_handler.vehicle_assignment[vehicle_id]
            VehicleHandler.add_new_trips(end_time, vehicle, trips, add=True)
        starting_time = end_time
