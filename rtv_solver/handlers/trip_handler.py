import numpy as np
import itertools
import multiprocessing as mp
import gurobipy as gp
import time
import traceback
from dataclasses import dataclass

from gurobipy import GRB

from rtv_solver.structure.trip_insertion_plan import TripInsertionPlan
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.config import Config
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.assignment_result import AssignmentResult

from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.network_handler import NetworkHandler


from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)


class TripHandler:
    """"
    TripHandler handles the generation of RTV combinations and eventually solves 

    The following items are tracked under class level to enable multiprocessing.\n
    TripHandler.tripCosts = []\n
    TripHandler.shared_trips_to_create []
    """
    def __init__(self,
                 vehicles: dict[int, Vehicle],
                 requests: list[Request],
                 active_requests: dict[float, Request],
                 iteration: int,
                 config: Config):
        self.config = config
        self.vehicles = vehicles
        self.requests = requests
        self.active_requests = active_requests
        self.iteration = iteration

        self.trips: list[Trip] = []     # basically collects all the tripCost objects for the feasible trips that are generated
        self.ondemand_only_trip_map = {}    # {request_id: trip_id}
        self.shared_trips_map = {}          # {cardinality: [shared_trip_id]}
        # vehicle<>trip mapping helper
        self.vehicle_to_trips_cost_map = {} # {vehicle_id: [trip_cost_index]}
        self.trip_to_vehicle_cost_map = {}  # {trip_id: [trip_cost_index]}
        
    def run(self):
        """
        Run trip generation: on-demand trips, trip costs, shared trips,
        Logs time spent on RTV generation.
        """
        self.starting_time = time.time()
        # TODO FIXME this does not count active vehicles # goal: if there is no more active vehicles, one can skip the iteration
        # TODO also does not need to run if we do not have any requests, does it?
        self.generate_ondemand_only_trips(self.requests, self.iteration)
        self.generate_trip_costs(self.vehicles, self.config.MAX_THREAD_CNT, 0)
        self.generate_shared_trips(self.vehicles, self.config.MAX_CARDINALITY, self.config.MAX_THREAD_CNT, self.config.SHARE_COST_FACTOR)

        console_logger.info(f"Time spent on RTV generation: {time.time() - self.starting_time:.3f} seconds. Number of trips generated: {len(self.trips)}.")
        return self.ondemand_only_trip_map, self.trips, TripHandler.trip_costs, self.vehicle_to_trips_cost_map, self.trip_to_vehicle_cost_map


    # SINGLE TRIP GENERATION
    def generate_ondemand_only_trips(self, requests: list[Request], iteration: int):
        """generate single trips from individual requests directly"""
        console_logger.info(f'{len(requests)} single trips generated from requests.')
        for request in requests:
            trip = self.create_trip_from_request(request, iteration, allow_walk=False)
            self.trips.append(trip)
            self.ondemand_only_trip_map[request.id] = trip.number

    def create_trip_from_request(self, request: Request, iteration: int, bus_combination=None,  first_last_mile_type=None, allow_walk: bool =True):
        if allow_walk and self._can_walk(request.origin, request.destination):
            return None # request can walk the entire way and does not need to be handled on, # TODO P10 add status that it was walkable and required no changes
        trip_no = self._get_new_trip_no()
        cost = self._get_trip_cost(request.origin, request.destination)
        return Trip.from_request(trip_no, request, iteration, cost, bus_combination, first_last_mile_type)

    @staticmethod
    def create_trip_for_picked_requests(boarded_requests: dict[str, Request], iteration) -> list[Trip]:
        # NOTE why do we not assign the corresponding vehicle here as well?
        trip_no = -1
        boarded_trips = []
        for request in boarded_requests.values():
            boarded_trips.append(Trip.from_request(trip_no, request, iteration))
            trip_no -=1
        return boarded_trips

    # TRIP COST GENERATION
    @staticmethod
    def create_trip_cost(vehicle: Vehicle, trip_no, trips, prev_sequence, config: Config):
        try:
            plan: TripInsertionPlan = VehicleHandler.plan_trip_insertions(vehicle, trips, prev_sequence=prev_sequence)
            feasible = plan.sequence_feasible
            if config.RETURN_DEPOT:
                feasible = plan.sequence_feasible and plan.depot_feasible
            if feasible:
                return TripCost(trip_no, vehicle.id, plan.added_cost, plan.sequence, plan)
            return None
        except Exception as e:
            console_logger.error(f"Error in create_trip_cost for trip {trip_no}, vehicle {vehicle.id}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    def _process_trip_cost_result(trip_cost: TripCost):
        if trip_cost != None:
            TripHandler.trip_costs.append(trip_cost)
    
    @staticmethod
    def _on_worker_error(e):
        console_logger.error("Worker crashed: %s", repr(e))
        console_logger.error("Error type: %s", type(e).__name__)
        console_logger.error("Error details: %s", str(e))
        traceback.print_exc()
        raise e

    def generate_trip_costs(self, vehicles, max_num_thread, trip_start):
        """
        generates trip costs for both single-request trips and shared multi-request trips to calculate the cost for each vehicle-trip combination
        
        Method applies mulitprocessing to run the TripCost generation in parallel.
        """
        if trip_start == 0: 
            TripHandler.trip_costs = []

        last_trip_cost_index = len(TripHandler.trip_costs)
        
        block_size = 1000 # number of trips handled in parallel
        for block_start in range(trip_start, len(self.trips), block_size):
            self._check_rtv_timeout()
            block_end = min(block_start + block_size, len(self.trips))
            # prepare arguments for multiprocessing
            pool = mp.Pool(max_num_thread)
            # iterate over all existing trips in blocks
            for trip in self.trips[block_start:block_end]:
                trips = []
                prev_trip_number = None
                if isinstance(trip, Trip):
                    trips = [trip]
                else: # instance - SharedTrip
                    shared_trip = trip
                    prev_trip_number = shared_trip.prev_trip_number
                    for sub_trip_no in shared_trip.trips:
                        sub_trip = self.trips[sub_trip_no] # NOTE what are subtrips?
                        trips.append(sub_trip)
                selected_vehicle_ids = vehicles.keys()
                if trip_start > 0: # only worth it when we have multiple trips to reduce combos TBC
                    selected_vehicle_ids = self._common_vehicles_of_trips(trips)
                # NOTE vehicle-loop seems to be only relevant for SharedTrips
                # TODO add documentation for how the different list and dicts interact
                for vehicle_id in selected_vehicle_ids:
                    # get previous sequence for the vehicle 
                    prev_sequence = []
                    if prev_trip_number is not None:
                        prev_costs = self.trip_to_vehicle_cost_map[prev_trip_number]
                        for prev_cost_index in prev_costs:
                            prev_trip_cost = TripHandler.trip_costs[prev_cost_index]
                            if prev_trip_cost.vehicle_id == vehicle_id:
                                prev_sequence = prev_trip_cost.sequence
                                break
                    pool.apply_async(
                        TripHandler.create_trip_cost, 
                        args=(vehicles[vehicle_id], trip.number, trips, prev_sequence, self.config), 
                        callback=TripHandler._process_trip_cost_result,
                        error_callback=TripHandler._on_worker_error)
            pool.close()
            pool.join()

        # >>> turn results into mappings for tracking 
        # initialize mappings Vehicle<>TripCost
        for vehicle_id in vehicles:
            if vehicle_id not in self.vehicle_to_trips_cost_map:
                self.vehicle_to_trips_cost_map[vehicle_id] = []
        for trip in self.trips[trip_start:]:
            if trip.number not in self.trip_to_vehicle_cost_map:
                self.trip_to_vehicle_cost_map[trip.number] = []

        # update mappings
        trip_cost_index = last_trip_cost_index
        for trip_cost in TripHandler.trip_costs[last_trip_cost_index:]:
            vehicle_id = trip_cost.vehicle_id
            trip_no = trip_cost.trip_no
            self.vehicle_to_trips_cost_map[vehicle_id].append(trip_cost_index)
            trip = self.trips[trip_no]
            if isinstance(trip, Trip):
                self.trip_to_vehicle_cost_map[trip_no].append(trip_cost_index)
            else: # instance: SharedTrip
                self.trip_to_vehicle_cost_map[trip_no].append(trip_cost_index)
                for sub_trip_no in trip.trips:
                    # NOTE check sub-trip generation
                    self.trip_to_vehicle_cost_map[sub_trip_no].append(trip_cost_index)
            trip_cost_index += 1

        console_logger.info(f"{len(TripHandler.trip_costs) - last_trip_cost_index} new trip costs generated.")

    # SHARED TRIP GENERATION
    @staticmethod
    def can_share_trips(prev_trip_no, trips, trip_nos, new_trip, current_cost, current_sequence, SHAREABLE_COST_FACTOR):
        feasible, cost, sequence = VehicleHandler.can_serve_trips(trips, new_trip, current_sequence)
        if feasible and cost <= SHAREABLE_COST_FACTOR * current_cost:
            return SharedTrip(prev_trip_no, 0, trip_nos, cost, sequence)
        return None
    
    @staticmethod
    def _process_shared_trip_result(shared_trip):
        if shared_trip != None:
            TripHandler.shared_trips_to_create.append(shared_trip)

    def _update_shared_trip_numbers(self, cardinality):
        # TODO add docstring
        for shared_trip in TripHandler.shared_trips_to_create:
            new_shared_trip_no = self._get_new_trip_no()
            shared_trip.number = new_shared_trip_no
            self.trips.append(shared_trip)
            self.shared_trips_map[cardinality].append(new_shared_trip_no)
            self.selected_combinations.append(shared_trip.trips)

    def _check_any_vehicles_available(self, trips: list[Trip]):
        """check whether there is any vehicles available for this set of trips"""
        return len(self._common_vehicles_of_trips(trips)) > 0
    
    def _common_vehicles_of_trips(self, trips):
        """per trip, collect all vehicles that can act on these trips and return that set of vehicle_indices"""
        common_vehicles = []
        for trip in trips:
            vehicles = []
            for trip_cost_index in self.trip_to_vehicle_cost_map[trip.number]:
                trip_cost = TripHandler.trip_costs[trip_cost_index]
                vehicles.append(trip_cost.vehicle_id)
            if len(common_vehicles) == 0:
                common_vehicles = set(vehicles)
            else:
                common_vehicles = common_vehicles.union(set(vehicles))
            if len(common_vehicles) == 0: # NOTE why is this condition here?
                return common_vehicles
        return common_vehicles

    def _create_rr_graph(self):
        """creates a matrix of two single trips (respectively requests) that can be shared"""
        # FIXME where does the KeyError come from in the second full iteration of a RH run?
        try: 
            self.rr_graph = {}
            for trip_no in self.ondemand_only_trip_map.values():
                self.rr_graph[trip_no] = set()
            for shared_trip_index in self.shared_trips_map[2]:
                shared_trip = self.trips[shared_trip_index]
                trip_no1, trip_no2 = shared_trip.trips
                self.rr_graph[trip_no1].add(trip_no2)
                self.rr_graph[trip_no2].add(trip_no1)
        except Exception as e:
            raise e

    def generate_shared_trips(self, vehicles, max_cardinality, max_num_thread, SHAREABLE_COST_FACTOR):
        """
        create all possible shared trips with as many of the initial requests up to max_cardinality.
        """
        cardinality = 2
        self.selected_combinations = []
        while cardinality <= max_cardinality:
            self._check_rtv_timeout()
            trip_start = len(self.trips)
            TripHandler.shared_trips_to_create = []
            # self.shared_trips_to_create = []
            st = time.time()
            self.shared_trips_map[cardinality] = []
            if cardinality == 2:
                no_of_trips = len(self.trips)
                pool = mp.Pool(max_num_thread)
                # create all possible combinations of trips with cardinality = 2
                for trip_nos in itertools.combinations(list(range(no_of_trips)), cardinality):
                    trip1 = self.trips[trip_nos[0]]
                    trip2 = self.trips[trip_nos[1]]
                    current_cost = trip1.cost + trip2.cost # get simple cost addition
                    trips = {}
                    for trip_no in trip_nos:
                        trip = self.trips[trip_no]
                        trips[trip.id] = trip
                    if self._check_any_vehicles_available(trips.values()):
                        pool.apply_async(
                            TripHandler.can_share_trips,
                            args=(trip_nos[0], trips, set(trip_nos), trip1, current_cost, [], SHAREABLE_COST_FACTOR,), 
                            callback=TripHandler._process_shared_trip_result,
                            error_callback=TripHandler._on_worker_error)
                pool.close()
                pool.join()
            else:
                tried_combinations = {}
                shared_trips_to_process = []
                prev_shared_trips = self.shared_trips_map[cardinality-1]
                block_size = 500
                console_logger.debug("Starting to process shared trips of cardinality {0}".format(cardinality))
                for shared_trip1_index in prev_shared_trips: # iterate only trips that already work for prior cardinality
                    shared_trip1 = self.trips[shared_trip1_index]
                    for request_id in self.ondemand_only_trip_map:
                        # check for timeout every block_size iterations
                        if len(shared_trips_to_process) % block_size == 0:
                            self._check_rtv_timeout()
                        
                        trip_no = self.ondemand_only_trip_map[request_id]
                        trip = self.trips[trip_no]

                        # Check if the trip is already part of the shared trip
                        if trip_no in shared_trip1.trips:
                            continue
                        # create the new trip_nos by adding the new one
                        trip_nos = shared_trip1.trips.copy()
                        trip_nos.add(trip_no)

                        # Check if this combination of trip numbers has already been tried
                        trips_signature = tuple(sorted(trip_nos))
                        if trips_signature in tried_combinations:
                            continue
                        tried_combinations[trips_signature] = 0

                        # if no cost was created prior, we can skip this particular request? NOTE why is that here? it should fail for all of them right?
                        if len(self.trip_to_vehicle_cost_map[shared_trip1_index]) == 0:
                            console_logger.debug(f"Trip<>Vehicle check {shared_trip1_index} failed with {request_id}")
                            continue

                        # Check if this trip can share a ride with all other trips in shared_trip1
                        rr_check_fail = False
                        for temp_trip_no in shared_trip1.trips:
                            if trip_no not in self.rr_graph[temp_trip_no]:
                                rr_check_fail = True
                                break
                            if temp_trip_no not in self.rr_graph[trip_no]:
                                rr_check_fail = True
                                break
                        if rr_check_fail:
                            continue
                        
                        # if there is an available vehicle, add cost from both trips; collect partial trips and check if vehicle is available
                        if self._check_any_vehicles_available([shared_trip1, trip]): # pre-check
                            current_cost = trip.cost + shared_trip1.cost
                            trips_collection = {}
                            for trip_no in trip_nos:
                                temp_trip = self.trips[trip_no]
                                trips_collection[temp_trip.id] = temp_trip
                            if self._check_any_vehicles_available(trips_collection.values()): # detailed check for specific trips
                                # create args for parallel execution but it is never used
                                args=(shared_trip1_index, trips_collection, trip_nos, trip, current_cost, shared_trip1.sequence, SHAREABLE_COST_FACTOR,)
                                shared_trips_to_process.append(args) # TODO we should be able to get rid of this as we do not parallelize? NOTE why do we not parallelize?
                                
                                # new_shared_trip = SharedTrip(shared_trip1_index, 0, trip_nos, current_cost, [])
                                # TripHandler._process_shared_trip_result(new_shared_trip)
                # console_logger.info(f"Number of shared trip combinations to process: {0}, time: {1}".format(len(shared_trips_to_process),time.time()-st))
                for block_start in range(0, len(shared_trips_to_process), block_size):
                    self._check_rtv_timeout()
                    block_end = min(block_start + block_size, len(shared_trips_to_process))
                    pool = mp.Pool(max_num_thread)
                    for args in shared_trips_to_process[block_start:block_end]:
                        pool.apply_async(TripHandler.can_share_trips, 
                                         args=args, 
                                         callback=TripHandler._process_shared_trip_result,
                                         error_callback=TripHandler._on_worker_error)
                    pool.close()
                    pool.join()
                console_logger.debug("Time to process shared trips of cardinality {0}: {1}".format(cardinality,time.time()-st))
            
            self._update_shared_trip_numbers(cardinality)
            
            if cardinality == 2: 
                self._create_rr_graph()
            if len(self.shared_trips_map[cardinality]) == 0:
                break # no trip_cost generation if no trips exist
            console_logger.info(f"{len(self.shared_trips_map[cardinality])} cardinality {cardinality} trips generated in {time.time()-st:.3f} seconds.")
            self.generate_trip_costs(vehicles, max_num_thread, trip_start)           
            cardinality += 1

    # INTERNAL HELPERS                         
    def _get_new_trip_no(self):
        return len(self.trips)

    def _check_rtv_timeout(self):
        time_spent = time.time() - self.starting_time
        if time_spent > self.config.RTV_TIMEOUT:
            raise Exception("RTV generation timedout: {0} > {1}".format(time_spent, self.config.RTV_TIMEOUT))

    def _can_walk(self, origin, destination):
        distance = NetworkHandler.travel_distance(origin, destination)
        return distance <= self.config.WALK_DISTANCE_CUTOFF
    
    @staticmethod
    def _get_trip_cost(origin, destination):
        return NetworkHandler.travel_distance(origin, destination)  
    
    # BELOW UNUSED METHODS
    def create_trip(self, request, am_capacity, wc_capacity, origin, destination, pick_up_time, latest_pick_up_time, earliest_arrival_time, latest_arrival_time, dwell_pickup, dwell_alight, iteration, bus_combination=None, first_last_mile_type=None, allow_walk=True):
        """create a trip from all the inputs"""
        if allow_walk and self._can_walk(origin, destination):
            return None
        trip_no = self._get_new_trip_no()
        cost = self._get_trip_cost(origin, destination)
        return Trip(
            request.id,
            trip_no,
            am_capacity, 
            wc_capacity, 
            pick_up_time, 
            latest_pick_up_time, 
            earliest_arrival_time,
            latest_arrival_time, 
            origin, 
            destination,
            dwell_pickup, 
            dwell_alight, 
            iteration, 
            cost,
            bus_combination=bus_combination,
            first_last_mile_type=first_last_mile_type)

    def get_first_mile_trip(self, request: Request, bustrip):
        origin = request.origin
        destination = bustrip.pick_up_stop_node
        return self.create_trip(
            request,
            origin,
            destination,
            request.earliest_pickup_time, 
            bustrip.leaving_time,
            bus_combination=bustrip.id,
            first_last_mile_type=0)

    def get_last_mile_trip(self, request: Request, bustrip):
        destination = request.destination
        origin = bustrip.destination_stop_node
        return self.create_trip(
            request,
            origin,
            destination,
            bustrip.arrival_time, 
            request.arrival_time,
            bus_combination=bustrip.id,
            first_last_mile_type=1)