import numpy as np
import logging
import itertools
import multiprocessing as mp
import gurobipy as gp
import time
import traceback

from gurobipy import GRB

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.config import Config
from rtv_solver.structure.vehicle import Vehicle

from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.network_handler import NetworkHandler

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
        self.trips: list[TripCost] = []
        self.ondemand_only_trip_map = {}    # {request_id: trip_id}
        self.shared_trips_map = {}          # {cardinality: [shared_trip_id]}
        # vehicle<>trip mapping helper
        self.vehicle_to_trips_cost_map = {} # {vehicle_id: [trip_cost_index]}
        self.trip_to_vehicle_cost_map = {}  # {trip_id: [trip_cost_index]}
        # gurobi assignment
        self.rebalancing_assignment = {}    # {vehicle_id: origin_node}
        self.vehicle_assignment = {}        # {vehicle_id: ([trips], StopSequence)}
        self.request_assignment = {}        # {request_id: vehicle_id}

        # self.selected_combinations = [] # NOTE why is it not initialized here?
        
        self.starting_time = time.time()
        if len(vehicles) != 0: 
            # TODO FIXME this does not count active vehicles # goal: if there is no more active vehicles, one can skip the iteration
            # TODO also does not need to run if we do not have any requests, does it?
            self.generate_ondemand_only_trips(requests, iteration)
            self.generate_trip_costs(vehicles, config.max_thread_cnt, 0)
            self.generate_shared_trips(vehicles, config.max_cardinality, config.max_thread_cnt, config.share_cost_factor)
            logging.info(f"Time spent on RTV generation: {time.time() - self.starting_time}")
            if len(TripHandler.trip_costs) > 0: 
                self.assign_trips_gurobi(requests, active_requests, config.ilp_penalty, config.keep_active)
                if config.rebalancing: # NOTE not sure if this should apply with trip_costs == 0; but it normally means that the vehicles are not in operation anymore
                    self.get_rebalancing_trips(vehicles,requests)
    
    # SINGLE TRIP GENERATION
    def generate_ondemand_only_trips(self, requests: list[Request], iteration: int):
        """generate single trips from individual requests directly"""
        logging.info(f'{len(requests)} single trips generated from requests.')
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
        plan = VehicleHandler.plan_trip_insertions(vehicle, trips, prev_sequence=prev_sequence)
        feasible = plan.sequence_feasible
        if config.return_depot:
            feasible = plan.sequence_feasible and plan.depot_feasible
        if feasible:
            return TripCost(trip_no, vehicle.id, plan.added_cost, plan.sequence)
        return None
    
    @staticmethod
    def _process_trip_cost_result(trip_cost):
        if trip_cost != None:
            TripHandler.trip_costs.append(trip_cost)
    
    @staticmethod
    def _on_worker_error(e):
        print("Worker crashed:", repr(e))
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

        logging.info(f"{len(TripHandler.trip_costs) - last_trip_cost_index} new trip costs generated.")

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
                logging.debug("Starting to process shared trips of cardinality {0}".format(cardinality))
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
                            logging.debug(f"Trip<>Vehicle check {shared_trip1_index} failed with {request_id}")
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
                # logging.info(f"Number of shared trip combinations to process: {0}, time: {1}".format(len(shared_trips_to_process),time.time()-st))
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
                logging.debug("Time to process shared trips of cardinality {0}: {1}".format(cardinality,time.time()-st))
            
            self._update_shared_trip_numbers(cardinality)
            
            if cardinality == 2: 
                self._create_rr_graph()
            if len(self.shared_trips_map[cardinality]) == 0:
                break # no trip_cost generation if no trips exist
            logging.info(f"{len(self.shared_trips_map[cardinality])} cardinality {cardinality} trips generated in {time.time()-st} seconds .")
            self.generate_trip_costs(vehicles, max_num_thread, trip_start)           
            cardinality += 1
    
    # FINAL ASSIGNMENT OF REQUEST-TRIP-VEHICLE combinations
    def assign_trips_gurobi(self, requests: list[Request], active_requests: dict[int, Request], penalty: int = 100_000, keep_active: bool = True):
        """
        ## ILP optimization of the previously generated trips to the possible vehicles

        :param list requests: Requests that are considered in this method call.
        :param list active_requests: Requests that have been accepted in prior iterations and that need to be kept based on the keep_active bool.
        :param int penalty: Changes the solution by penalizing requests that are not accepted.
        :param bool keep_active: To get the actual best result, we do not care about what has been previously accepted. Prior iterations only influence the solutions by already boarded requests. If a new combination becomes better, we do not want to be constrained by trips that have been accepted because the solver saw only a partial (earlier picture) and it should not be stuck with previously selected requests.

        NOTE If no active vehicles are available, it will still output logs as no optimization is possible. This only occurs if the vehicles are basically offline as the end_time of their operation has finished.
        """
        trip_count = len(TripHandler.trip_costs)
        request_count = len(requests)

        logging.debug("Started building optimization problem")
        # setup Integer Linear Program 
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model('RTV assignment - Service rate + Minimum distance', env=env)
            m.Params.OutputFlag = 0
            
            # define trip variables with related costs
            trip_costs_obj = np.fromiter((tc.cost for tc in TripHandler.trip_costs), dtype=float, count=trip_count)
            x_t = m.addVars(trip_count,
                            lb=0,
                            ub=1,
                            obj=trip_costs_obj,
                            name="t", 
                            vtype=GRB.BINARY)

            # create penalties per request
            request_ids = np.array([r.id for r in requests])
            priorities  = np.array([r.priority for r in requests])
            penalties = priorities.copy()
            if keep_active:
                penalties[np.isin(request_ids, list(active_requests))] = 100
            x_r = m.addVars(request_count,
                            lb=0,
                            ub=1,
                            obj=penalties * penalty,
                            name="r", 
                            vtype=GRB.BINARY)

            # constraint: each vehicle has at most on trip
            m.addConstrs((gp.quicksum(x_t[i] for i in self.vehicle_to_trips_cost_map[vehicle_id]) <= 1 for vehicle_id in list(self.vehicle_to_trips_cost_map.keys())), "veh")

            # constraint: each request is either rejected or served by a single trip 
            # active requests are handled with extra care
            request_no = 0
            for request in requests:
                trip_no = self.ondemand_only_trip_map[request.id]
                cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]

                m.addConstr(x_r[request_no]+gp.quicksum(x_t[i] for i in cost_map_indices) == 1, "req_{0}".format(request.id))
                
                # all the previously assigned requests should be picked up
                if request.id in active_requests and keep_active:
                    m.addConstr(x_r[request_no] == 0, "active_req_{0}".format(request.id))
                request_no+=1

            m.setParam('TimeLimit', self.config.ilp_timeout)
            m.optimize()

            # # extract solution from Gurobi assignment
            self.trip_sizes = []
            self.unassigned_trip_count = 0
            self.taxi_only_trip_count = 0
            self.with_one_bus_trip_count = 0
            self.with_two_bus_trip_count = 0
            self.added_distance = 0

            if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
                logging.info("Total time spent on optimization: {0}".format(m.Runtime))

                for vehicle_id in self.vehicle_to_trips_cost_map:
                    for i in self.vehicle_to_trips_cost_map[vehicle_id]:
                        if x_t[i].X == 1:
                            trip_cost = TripHandler.trip_costs[i]
                            self.added_distance += trip_cost.cost
                            trip_no = trip_cost.trip_no
                            trip = self.trips[trip_no]
                            trips = []
                            if isinstance(trip,Trip):
                                trips.append(trip)
                            else:
                                for sub_trip_no in trip.trips:
                                    trips.append(self.trips[sub_trip_no])
                            self.trip_sizes.append(len(trips))
                            self.vehicle_assignment[vehicle_id] = (trips, trip_cost.sequence)
                            logging.info(f"Assignment: {trip_cost}")
                            # print(f"Assign: veh-{vehicle_id} with trips {[trip.id for trip in trips]} under cost {trip_cost.cost} with sequence {TripCost.sequence_to_str(trip_cost.sequence)}")

                for request in requests:
                    found_assignment = False
                    trip_no = self.ondemand_only_trip_map[request.id]
                    cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
                    for index in cost_map_indices:
                        if x_t[index].X == 1:
                            trip_cost = TripHandler.trip_costs[index]
                            vehicle_id = trip_cost.vehicle_id
                            self.request_assignment[request.id] = vehicle_id
                            found_assignment = True
                            self.taxi_only_trip_count +=1
                            break

                    if not found_assignment:
                        self.unassigned_trip_count += 1
            else:
                self.unassigned_trip_count = request_count
                # Compute IIS (conflicting constraints)
                m.Params.OutputFlag = 1
                m.computeIIS()
                m.write("infeasible.ilp")   # human-readable
                m.write("infeasible.lp")    # full model
                m.write("infeasible.mps")   # optional

                # Print which constraints are in IIS
                print("\n--- IIS constraints ---")
                for constraint in m.getConstrs():
                    if constraint.IISConstr:
                        print("IIS:", constraint.ConstrName)
                raise Exception(f"Gurobi solver ended with code: {m.Status}") # Code 3 INFEASIBLE
                        
            logging.info(f'Assignment: new requests / unassigned / assigned: {request_count} / {self.unassigned_trip_count} / {self.taxi_only_trip_count}')
            # TODO make information better, some requests are re-assigned although they were already assigned

    def get_rebalancing_trips(self, vehicles, requests):
        """ 
        Idling, empty vehicles can be reallocated to the origins of current requests that have not been covered
        
        Assumption: rejected requests are in underserved areas, so we need to send additional vehicles there. 
        """
        empty_vehicles = []
        for vehicle_id in vehicles:
            if vehicle_id not in self.vehicle_assignment:
                vehicle = vehicles[vehicle_id]
                if not (vehicle.rebalancing or len(vehicle.stop_sequence)>0):
                    empty_vehicles.append(vehicle_id)
                
        unassigned_requests = []
        for request in requests:
            if request.id not in self.request_assignment:
                unassigned_requests.append(request)

        number_of_vehicles = len(empty_vehicles)
        number_of_requests = len(unassigned_requests)
        max_rebalancing_count = min(number_of_vehicles, number_of_requests)

        if max_rebalancing_count > 0:
            m = gp.Model('Rebalancing')
            rebalancing_costs = np.zeros((number_of_vehicles,number_of_requests))
            for i in range(number_of_vehicles):
                vehicle = vehicles[empty_vehicles[i]]
                for j in range(number_of_requests):
                    # get origin for unassigned requests and costs for idling vehicles to get to that position
                    origin = unassigned_requests[j].origin
                    rebalancing_costs[i][j] = VehicleHandler.cost_of_rebalancing(vehicle, origin)
            y_vr = m.addVars(number_of_vehicles, number_of_requests, lb=0, ub=1, obj=rebalancing_costs, name="y_vr", vtype=GRB.BINARY)

            m.addConstrs((y_vr.sum(i,'*') <= 1 for i in range(number_of_vehicles)), "veh")
            m.addConstrs((y_vr.sum('*',j) <= 1 for j in range(number_of_requests)), "req")
            m.addConstr((y_vr.sum() <= max_rebalancing_count), "total_assignment")
            m.optimize()

            # process optimization result
            if m.Status == GRB.OPTIMAL:
                for i in range(number_of_vehicles):
                    for j in range(number_of_requests):
                        if y_vr[i,j].X == 1:
                            vehicle_id = empty_vehicles[i]
                            origin = unassigned_requests[j].origin
                            self.rebalancing_assignment[vehicle_id] = origin
                            break

    # INTERNAL HELPERS                         
    def _get_new_trip_no(self):
        return len(self.trips)

    def _check_rtv_timeout(self):
        time_spent = time.time() - self.starting_time
        if time_spent > self.config.rtv_timeout:
            raise Exception("RTV generation timedout: {0} > {1}".format(time_spent, self.config.rtv_timeout))

    def _can_walk(self, origin, destination):
        distance = NetworkHandler.travel_distance(origin, destination)
        return distance <= self.config.walk_distance_cutoff
    
    @staticmethod
    def _get_trip_cost(origin, destination):
        return NetworkHandler.travel_distance(origin, destination)  
    
    # BELOW UNUSED METHODS
    def create_trip(self, request, am_capacity, wc_capacity, origin, destination, pick_up_time, latest_pick_up_time, earliest_arrival_time, latest_arrival_time, dwell_pickup, dwell_alight, iteration, bus_combination=None, first_last_mile_type=None, allow_walk=True):
        if allow_walk and self._can_walk(origin, destination):
            return None
        trip_no = self._get_new_trip_no()
        cost = self._get_trip_cost(origin,destination)
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