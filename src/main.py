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

BASE_DATA_DIR = "../data/"
BATCH_INTERVAL = timedelta(0,seconds=30)
BUS_DWELL = 25 # second
AVERAGE_EDGE_SPEED = 10.785 #mps
WALK_DISTANCE_CUT_OFF = 500 #meters
BUS_DISTANCE_CUT_OFF = 1000 #meters
IPM_SOLVER_TIMEOUT = 1200
PENALTY = 1000000
MAX_WAIT_TIME = 300 # 5 minutes
ADDITIONAL_TRIP_TIME_FACTOR = 1.5 #we allow trips to have 2 x shortest path travel distance
SHAREABLE_COST_FACTOR = 1

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Simulator arguments')
    parser.add_argument('--max_number_of_vehicles', type=int,
                    help='maximum number of MoD vehicles')
    parser.add_argument('--max_capacity', type=int,
                    help='maximum capacity of a MoD vehicle')
    parser.add_argument('--allow_bus_transfer', action=argparse.BooleanOptionalAction,
                    help='allow bus transfers')
    parser.add_argument('--allow_bus', action=argparse.BooleanOptionalAction,
                    help='allow busses')
    parser.add_argument('--out_put_dir',
                    help='output directory')
    args = parser.parse_args()
    OUTPUT_DIR = args.out_put_dir

    logging.basicConfig(filename=OUTPUT_DIR+'main.log', level=logging.DEBUG)
    logging.info('Starting the simulator with: max_number_of_vehicles {0}, max_capacity {1}'.format(args.max_number_of_vehicles, args.max_capacity))
    exe_start_time=time.time()
    iteration = 0
    network_handler = NetworkHandler(BASE_DATA_DIR+"new_map/")

    request_handler = RequestHandler(BASE_DATA_DIR+"requests/requests.csv",MAX_WAIT_TIME,ADDITIONAL_TRIP_TIME_FACTOR)
    starting_time = request_handler.earliest_start_time(network_handler)
    latest_time = request_handler.latest_start_time(network_handler)
    start_of_the_day = starting_time.replace(hour=0, minute=0, second=0, microsecond=0)
    bus_handler = BusHandler(BASE_DATA_DIR+"bus/", start_of_the_day, BUS_DWELL, network_handler, AVERAGE_EDGE_SPEED, BUS_DISTANCE_CUT_OFF)
    vehicle_handler = VehicleHandler(BASE_DATA_DIR+"vehicles/vehicles.csv",start_of_the_day,AVERAGE_EDGE_SPEED,args.max_number_of_vehicles, args.max_capacity)

    # OUTPUT_DIR = "../output/"
    output_handler = OutputHandler(OUTPUT_DIR)

    while starting_time <= latest_time:
        end_time = starting_time + BATCH_INTERVAL
        batch = request_handler.get_batch(network_handler,starting_time,end_time)
        request_bus_combinations = {}
        for request in batch:
            request_bus_combinations[request.id] = bus_handler.generate_bus_trips(network_handler, request, args.allow_bus, args.allow_bus_transfer)

        trip_handler = TripHandler(end_time,network_handler,vehicle_handler,batch,request_bus_combinations,AVERAGE_EDGE_SPEED, WALK_DISTANCE_CUT_OFF,IPM_SOLVER_TIMEOUT,PENALTY,2,SHAREABLE_COST_FACTOR)
        output_handler.record_output(end_time,trip_handler)
        for vehicle_id in trip_handler.vehicle_assignment:
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips = trip_handler.vehicle_assignment[vehicle_id]
            vehicle_handler.add_new_trips(network_handler,end_time, vehicle, trips, add=True)
        starting_time = end_time
