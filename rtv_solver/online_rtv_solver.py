import copy
import multiprocessing
import sys
from multiprocessing import Pool
import time
import numpy as np
import copy

from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.trip_handler import TripHandler, RTVTimeoutError
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.swap_handler import SwapHandler

from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.config import Config
from rtv_solver.structure.driver_run import DriverRun, ManifestEntry

from rtv_solver.pipeline import CO_TripCostMinimization, CO_RebalancingCoverage

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class ManifestConsistencyError(Exception):
    """Raised when a manifest consistency check fails."""

class OnlineRTVSolver:
    """solves entire RTV problem for a given payload """
    def __init__(self, config: Config = None):
        self.config = config
        if sys.platform == "darwin": # required to run online_solver correctly on MacOS
            try:
                multiprocessing.set_start_method("fork")
            except RuntimeError: # start method was already set somewhere else -> don't crash
                pass

    def check_feasibility(self, payload):
        # NOTE for what do we need this method?
        NetworkHandler.init_from_source(server_url=self.config.SERVER_URL)
        feasible_time_slots = []

        request = payload[PayloadParser.REQUESTS][0]
        pickup_pt, dropoff_pt = request[PayloadParser.REQ_PICKUP_PT], request[PayloadParser.REQ_DROPOFF_PT]
        origin = Node(pickup_pt["lat"], pickup_pt["lon"])
        destination = Node(dropoff_pt["lat"], dropoff_pt["lon"])
        request_travel_time = NetworkHandler.travel_time(origin, destination)

        for time_window in request["time_windows"]: # NOTE where does the 'time_windows' come from? and the strings below are thus not changed as we do not know its origin
            request_copy = copy.deepcopy(request)
            request_copy[PayloadParser.REQ_PICKUP_WINDOW_START] = time_window["pickup_time_window_start"]
            request_copy[PayloadParser.REQ_PICKUP_WINDOW_END] = time_window["pickup_time_window_end"]
            request_copy[PayloadParser.REQ_DROPOFF_WINDOW_START] = time_window["dropoff_time_window_start"]
            request_copy[PayloadParser.REQ_DROPOFF_WINDOW_END] = time_window["dropoff_time_window_end"]
            best_cost = float("inf")
            for driver_run in payload[PayloadParser.DRIVERS]:
                cost, _ = self._insert_request_to_driver_run(
                    payload[PayloadParser.DEPOT], driver_run, request_copy)
                if cost >= 0 and cost < best_cost:
                    best_cost = cost
            if best_cost < float("inf"):
                feasible_time_slots.append((time_window,best_cost / request_travel_time))

        return feasible_time_slots

    def solve_pdptw_rtv(self, payload, iteration = 0): # TODO do we need to add current_time
        """
        Solver for the entire payload.
        
        With conifg.return_depot, this method will not add the final trips to the depot. The user has to call finalize_driverRuns(...) to add those final stops. 
        """
        # initalize network and payload
        NetworkHandler.init_from_source(server_url=self.config.SERVER_URL)
        payload_object = PayloadParser.get_payload_object(payload)
        # get all requests of payload, add 
        request_handler = RequestHandler(payload_object.requests, self.config.DWELL_PICKUP, self.config.DWELL_ALIGHT)
        temp_batch = request_handler.get_all_requests()
        
        # filter active and boarded requests for subsequent action as they need to be integrated when handling new trip generation
        request_batch = []
        active_requests = {}
        boarded_requests = {}
        for req in temp_batch:
            req_id = req.id
            if req_id in payload_object.boarded_requests_keys:
                boarded_requests[req_id] = req
            else:
                if req_id in payload_object.active_requests_keys:
                    active_requests[req_id] = req
                request_batch.append(req)
        
        # initialize all vehicles as they are stored in the payload-object
        vehicle_handler = VehicleHandler(payload_object.depot, 
                                         payload_object.driver_runs,
                                         self.config)
        # create trips of all already boarded requests # TODO these are not always boarded at this point
        boarded_trips = TripHandler.create_trip_for_picked_requests(boarded_requests, iteration)
        # update vehicle position/trips/times along its path according to all data stored in the manifest
        vehicle_handler.add_manifest_to_vehicles(payload_object.driver_runs,
                                                 boarded_requests,
                                                 boarded_trips, 
                                                 self.config.DWELL_ALIGHT, 
                                                 self.config.DWELL_PICKUP)

        # TODO add depot node with id, so it can be calculated later on using the travel_time_matrix
        
        NetworkHandler.initialize_travel_time_matrix() 
        iteration += 1  # increase iteration as the prior step was just rebuilding from the last iteration (if there was a prior step)
        
        try:
            # generate and assign trips to each vehicle using the RTV approach solved by an ILP
            trip_handler = TripHandler(
                vehicle_handler.vehicles,
                request_batch,
                active_requests,
                iteration,
                self.config)
            single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map = trip_handler.run()
            for trip_cost in trip_costs:
                console_logger.info(f"trip cost: {trip_cost.trip_no}, {trip_cost.cost}, {trip_cost.get_request_order_str()}")
        except RTVTimeoutError as e:
            console_logger.error(f"Error in trip generation: {e}")
            raise e
        
        if len(vehicle_handler.vehicles) != 0:
            
            optimizer = CO_TripCostMinimization(self.config)
            optimizer.reset(single_trip_map, 
                            trip_list, 
                            trip_costs, 
                            vehicle_to_trips_cost_map, 
                            trip_to_vehicle_cost_map )
            result = optimizer.run(request_batch, active_requests)

            # TODO add rebalancing handling including the assignment itself and validate its correct behavior
            # if self.config.REBALANCING:
            #     rebalancing_optimizer = CO_RebalancingCoverage(self.config)
            #     result = rebalancing_optimizer.run(result, vehicle_handler.vehicles, request_batch)
   
        # assign vehicles and add trips / sequence to each vehicle 
        unserved_requests = set([req.id for req in request_batch]) # number of requests that are not already confirmed to be  served
        for vehicle_id in result.vehicle_assignment: # if it is empty the assignment is skipped
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips, prev_sequence = result.vehicle_assignment[vehicle_id]
            plan = VehicleHandler.plan_trip_insertions(vehicle, trips, prev_sequence=prev_sequence)
            vehicle.apply_trip_insertion(plan)
            for trip in trips: # remove assigned trips from unserved
                if trip.request_id in unserved_requests:
                    unserved_requests.remove(trip.request_id)

        # update driver runs
        updated_driver_runs = []
        for driver_run in payload_object.driver_runs:
            new_driver_run = vehicle_handler.update_run(driver_run)
            # new_driver_run = self.update_run(vehicle_handler, driver_run)
            updated_driver_runs.append(new_driver_run)
            
        # check invariants whether manifest is still correct
        self._check_consistency_of_manifests(payload[PayloadParser.DRIVERS], 
                                            updated_driver_runs,
                                            unserved_requests, 
                                            payload[PayloadParser.REQUESTS],
                                            keep_active=self.config.KEEP_ACTIVE,
                                            return_depot=self.config.RETURN_DEPOT)
        # TODO remove this complex data structure to move data up the stack; integrate JSON logger that records information to a standardized JSON file and can later be rebuilt and analyzed using that JSON file (probably requires something similar to OutputHandler that was used in the simulation)
        assignment_status = {PayloadParser.STATS_ASSIGNED: result.request_assignment, 
                            PayloadParser.STATS_UNSERVED: list(unserved_requests)}
        return updated_driver_runs, assignment_status # ,trip_handler, vehicle_handler, request_handler, payload_object

    @staticmethod
    def _check_consistency_of_manifests(prev_driver_runs: list[dict], 
                                        new_driver_runs: list[dict], 
                                        unserved_requests: set[int], 
                                        new_requests: list[dict],
                                        keep_active: bool = False,
                                        check_depot: bool = False,
                                        return_depot: bool = False):
        """ 
        Checks whether the previously written manifest still aligns with the new manifest after new requests have been assigned. This concerns previously boarded or selected active requests. Each requests must be picked up AND dropped off exactly once and unserved_requests should not appear in the manifests at all.

        If config.keep_active = True - boarded and previously active requests are still part of the manifest.
        If config.keep_active = False - boarded requests must always remain, active requests can but need not be discarded.
        
        If check_depot = True, checks the depot conditions but not required during the assignment of trips.
        If config.return_depot = True - each vehicle needs a depot return trip in their manifest.
        If config.return_depot = False - each vehicle should not have a depot return.
        """
        # TODO test why are the booking_id np.float here? they must be changed in some position
        # required to check whether all items have 1 pickup and 1 dropoff
        picked_requests = set([req["booking_id"] for req in new_requests])
        dropped_requests = copy.deepcopy(picked_requests)
        # required to correctly consider the active or boarded condition in the previous manifest
        active_requests = set()
        boarded_requests = set()
        for driver_run in prev_driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            serviced_locations = state[PayloadParser.DRIVER_STATE_LOC_SERV] # vehicle state condition for active / boarding
            for stop in manifest:
                stop_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                action = stop[PayloadParser.MANIFEST_ACTION]
                stop_order = stop[PayloadParser.MANIFEST_ORDER]
                if action == VehicleStop.ACT_PICKUP:
                    if stop_order <= serviced_locations: # i.e., already boarded or finished
                        boarded_requests.add(stop_id)
                    else: 
                        active_requests.add(stop_id)
                    picked_requests.add(stop_id)
                elif action == VehicleStop.ACT_DROPOFF:
                    dropped_requests.add(stop_id) 

        # depending on config.keep_active, active requests also need to be part of the new_driver_runs
        # remove all requests that are picked up/dropped off (ensures that all stops exist twice) and track depot runs
        depot_requests = []
        manifests = []
        for driver_run in new_driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            manifests.append(manifest)

            for stop in manifest:
                stop_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                action = stop[PayloadParser.MANIFEST_ACTION]
                if action == VehicleStop.ACT_PICKUP:
                    boarded_requests.discard(stop_id)
                    if keep_active: # only additionally remove active requests when keep_active = True
                        active_requests.discard(stop_id)
                    picked_requests.remove(stop_id)
                elif action == VehicleStop.ACT_DROPOFF:
                    dropped_requests.remove(stop_id)
                elif action == VehicleStop.ACT_DEPOT:
                    depot_requests.append(stop_id) # stop_id should be -(vehicle.id+1) as defined in VehicleHandler.get_manifest()
        # remove all requests that are unserved            
        for req_id in unserved_requests:
            req_id = req_id
            picked_requests.remove(req_id)
            dropped_requests.remove(req_id)

        if len(boarded_requests) > 0:
            raise ManifestConsistencyError(f"ManifestError: boarded_requests {boarded_requests} were not picked up.")
        if keep_active and len(active_requests) > 0:
            raise ManifestConsistencyError(f"ManifestError: active_requests {active_requests} were not kept in the manifest despite config.keep_active = True.")
        if len(picked_requests) > 0 or len(dropped_requests) > 0:
            # TODO: fails with wilson_data, cardinality = 3, thread_cnt = 16, batch_interval = 1800, step_size = 1800 (should be reproducible with this)
            raise ManifestConsistencyError(f"ManifestError: Some requests could not be removed. Missing requests: {picked_requests}, {dropped_requests}")
        active_manifests = sum(1 for lst in manifests if lst) # compare against already active manifests
        if check_depot: 
            if return_depot:
                if len(depot_requests) != active_manifests: # as many depot returns as vehicles are running
                    raise ManifestConsistencyError(f"ManifestError: Depot return was not added, instead depot_returns {depot_requests} exist.") # debug: more or less than expected?
            else:
                if len(depot_requests) != 0:
                    raise ManifestConsistencyError(f"ManifestError: {len(depot_requests)} depot return was added despite self.config.return_depot = False.")
        return True

    @staticmethod
    def simulate_manifest(config: Config, current_time, driver_runs, intermediate_location=True, tt_matrix: np.ndarray = None):
        """
        Update the driver_run for the offlineSolver so that the last results fits the time that we have used for it.
        """
        NetworkHandler.init_from_source(server_url=config.SERVER_URL, tt_matrix=tt_matrix)
        new_driver_runs = []
        # TODO longterm: turn driver_run into an object that handles all the conditions and changes based on validated calls
        for driver_run in driver_runs:
            # get data from vehicle
            state = driver_run[PayloadParser.DRIVER_STATE]
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            next_immediate_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
            next_immediate_loc = state[PayloadParser.DRIVER_STATE_LOC]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            
            # update time if manifest is already completed
            if len(manifest) == current_order and next_immediate_time < current_time:
                next_immediate_time = current_time

            while len(manifest) > current_order and current_time >= manifest[current_order][PayloadParser.MANIFEST_SCHED_TIME]:
                next_stop = manifest[current_order]
                next_immediate_time = next_stop[PayloadParser.MANIFEST_SCHED_TIME]
                next_immediate_loc = next_stop[PayloadParser.MANIFEST_LOC]
                action = next_stop[PayloadParser.MANIFEST_ACTION]
                # apply dwell time if applicable, why do we have this 
                if action == VehicleStop.ACT_PICKUP:
                    next_immediate_time += config.DWELL_PICKUP
                elif action == VehicleStop.ACT_DROPOFF:
                    next_immediate_time += config.DWELL_ALIGHT
                else: # no extra time for depot or rebalance
                    next_immediate_time += 0
                current_order += 1
                if next_immediate_time > current_time:
                    break
                
            if len(manifest) > current_order and next_immediate_time < current_time and intermediate_location:
                # if manifest is longer than final stop (according to time limit) AND next_immediate_time is still smaller than current_time AND we want to have the location in between stops, we will get that location here
                next_immediate_node = NetworkHandler.get_node_from_manifest_location(next_immediate_loc)
                target_node = NetworkHandler.get_node_from_manifest_location(manifest[current_order][PayloadParser.MANIFEST_LOC])
                next_immediate_time, next_immediate_node = NetworkHandler.get_current_location_time(
                    next_immediate_node, target_node, next_immediate_time, current_time)
                next_immediate_loc = {"lat":next_immediate_node.lat,
                                      "lon":next_immediate_node.lon,
                                      "node_id":next_immediate_node.id}

            state[PayloadParser.DRIVER_STATE_DT_SEC] = next_immediate_time
            state[PayloadParser.DRIVER_STATE_LOC] = next_immediate_loc
            state[PayloadParser.DRIVER_STATE_LOC_SERV] = current_order
            new_driver_runs.append({
                PayloadParser.DRIVER_STATE: state,
                PayloadParser.DRIVER_MANIFEST: manifest})
        
        OnlineRTVSolver._check_consistency_of_manifests(driver_runs, new_driver_runs, [], [], keep_active=config.KEEP_ACTIVE, return_depot=config.RETURN_DEPOT)
        return new_driver_runs

    def solve_pdptw_heuristic(self, payload, return_added_vmt=False):
        """ uses heuristic to
        DO NOT USE IT FOR RTV solution. 
        """
        updated_driver_runs = copy.deepcopy(payload[PayloadParser.DRIVERS])
        total_cost = 0
        unserved_requests = []
        for request in payload[PayloadParser.REQUESTS]:
            cheapest_vehicle = None
            cheapest_cost = float("inf")
            cheapest_vehicle_index = -1
            for vehicle_index in range(len(updated_driver_runs)):
                driver_run = updated_driver_runs[vehicle_index]
                cost, new_driver_run = self._insert_request_to_driver_run(payload[PayloadParser.DEPOT], driver_run, request)
                if cost >=0 and cost < cheapest_cost:
                    cheapest_cost = cost
                    cheapest_vehicle = new_driver_run
                    cheapest_vehicle_index = vehicle_index
            if cheapest_vehicle is not None:
                updated_driver_runs[cheapest_vehicle_index] = cheapest_vehicle
                total_cost += cheapest_cost
            else:
                unserved_requests.append(request[PayloadParser.REQ_BOOKING_ID])
        
        self._check_consistency_of_manifests(payload[PayloadParser.DRIVERS], updated_driver_runs, unserved_requests, payload[PayloadParser.REQUESTS], keep_active=self.config.KEEP_ACTIVE, return_depot=self.config.RETURN_DEPOT)
        if return_added_vmt:
            return updated_driver_runs, unserved_requests, total_cost
        return updated_driver_runs, unserved_requests

    def solve_pdptw(self, payload, skip_swapping=True):
        """
        takes payload and decides which solver to run (heuristic or RTV)
        """
        # NOTE what is the difference to PDPTW_RTV
        # TODO currently not working due to the changes of the return values of solve-pdptw-rtv
        remaining_requests = []
        for driver_run in payload[PayloadParser.DRIVERS]:
            current_order = driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_LOC_SERV]
            remaining_manifest = driver_run[PayloadParser.DRIVER_MANIFEST][current_order:]
            unique_requests = set()
            for stop in remaining_manifest:
                booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                if booking_id not in unique_requests:
                    unique_requests.add(booking_id)
            remaining_requests.append(len(unique_requests))
        
        remaining_requests = np.array(remaining_requests)
        if remaining_requests.max() <= self.config.MAX_CARDINALITY:
            updated_driver_runs, unserved_requests = self.solve_pdptw_rtv(payload)
            if len(unserved_requests) == 0:
                return updated_driver_runs, unserved_requests

        # Use heuristic if any vehicle has too many remaining requests
        console_logger.debug("Inserting with heuristic...")
        # Get the initial solution with insertion heuristic
        updated_driver_runs, unserved_requests = self.solve_pdptw_heuristic(payload)
        if len(unserved_requests) > 0:
            console_logger.debug("Unserved requests after heuristic: %d", len(unserved_requests))
            # Return without further optimization if there are unserved requests
            return updated_driver_runs, unserved_requests

        if skip_swapping:
            return updated_driver_runs, unserved_requests
        # If all requests are served, try to optimize the solution further
        console_logger.debug("Optimizing solution with swap heuristic...")
        start_time = time.time()
        swap_handler = SwapHandler(self.config.SERVER_URL,
                                   updated_driver_runs,
                                   payload[PayloadParser.DEPOT],
                                   config=self.config)
        swaped_driver_runs, reduced_cost, no_of_swaps = swap_handler.run_swap()
        while no_of_swaps > 0 and reduced_cost > 0 and time.time() - start_time < self.RTV_TIMEOUT:
            updated_driver_runs = swaped_driver_runs
            swaped_driver_runs, reduced_cost, no_of_swaps = swap_handler.run_swap(rerunning=True)

        return swaped_driver_runs, unserved_requests

    @staticmethod
    def _evaluate_insertion(args):
        """ accepts a single set of args and evaluates the benefit or cost of the insertion into the existing route """
        i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective, depot, end_time, dwell_pickup, dwell_alight = args
        new_manifest = copy.deepcopy(remaining_stops[:i] + [pickup_stop] + remaining_stops[i:j] + [dropoff_stop] + remaining_stops[j:])
        current_time = start_time
        current_node = start_node
        current_load = load
        cost = 0
        order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        index = 0
        for stop in new_manifest:
            stop_location = stop[PayloadParser.MANIFEST_LOC]
            next_node = Node(stop_location["lat"], 
                             stop_location["lon"],
                             id = stop_location["node_id"])
            travel_time = NetworkHandler.travel_time(current_node, next_node)
            cost += travel_time
            current_node = next_node
            current_time += travel_time
            if current_time < stop[PayloadParser.MANIFEST_TIME_WINDOW_START]:
                current_time = stop[PayloadParser.MANIFEST_TIME_WINDOW_START]
            stop[PayloadParser.MANIFEST_SCHED_TIME] = current_time
            if objective == "pick_up_time" and (i == index or j == index):
                stop[PayloadParser.MANIFEST_TIME_WINDOW_END] = current_time + 30
            if current_time > stop[PayloadParser.MANIFEST_TIME_WINDOW_END]:
                return float("inf"), None
            if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                current_load += stop[PayloadParser.MANIFEST_AMBULATORY]
                current_time += dwell_pickup
            else:
                current_load -= stop[PayloadParser.MANIFEST_AMBULATORY]
                current_time += dwell_alight
            if current_load > state[PayloadParser.DRIVER_STATE_AM_CAP]:
                return float("inf"), None
            order += 1
            stop[PayloadParser.MANIFEST_ORDER] = order
            index += 1

        if current_time + NetworkHandler.travel_time(current_node,depot) > end_time:
            return float("inf"), None
        if objective == "pick_up_time":
            return new_manifest[i][PayloadParser.MANIFEST_SCHED_TIME], new_manifest
        return cost, new_manifest

    def _insert_request_to_driver_run(self, depot, driver_run, request, objective="vmt"):
        NetworkHandler.init_from_source(server_url=self.config.SERVER_URL)
        driver_run_c = copy.deepcopy(driver_run)
        depot_pt = depot[PayloadParser.DEPOT_PT]
        depot_node_id = NetworkHandler.get_next_node_id(depot_pt["lat"], depot_pt["lon"])
        depot_node = Node(
            depot_pt["lat"], 
            depot_pt["lon"], 
            id =depot_node_id)

        pickup_stop = {
            PayloadParser.MANIFEST_RUN_ID: None, 
            PayloadParser.MANIFEST_BOOKING_ID: request[PayloadParser.REQ_BOOKING_ID], 
            PayloadParser.MANIFEST_ORDER: -1, 
            PayloadParser.MANIFEST_ACTION: VehicleStop.ACT_PICKUP, 
            PayloadParser.MANIFEST_LOC: request[PayloadParser.REQ_PICKUP_PT], 
            PayloadParser.MANIFEST_SCHED_TIME: -1, 
            PayloadParser.MANIFEST_AMBULATORY: request[PayloadParser.REQ_AMBULATORY], 
            PayloadParser.MANIFEST_WHEELCHAIR: request[PayloadParser.REQ_WHEELCHAIR], 
            PayloadParser.MANIFEST_TIME_WINDOW_START: request[PayloadParser.REQ_PICKUP_WINDOW_START],
            PayloadParser.MANIFEST_TIME_WINDOW_END: request[PayloadParser.REQ_PICKUP_WINDOW_END]}
        dropoff_stop = {
            PayloadParser.MANIFEST_RUN_ID: None, 
            PayloadParser.MANIFEST_BOOKING_ID: request[PayloadParser.REQ_BOOKING_ID], 
            PayloadParser.MANIFEST_ORDER: -1, 
            PayloadParser.MANIFEST_ACTION: VehicleStop.ACT_DROPOFF, 
            PayloadParser.MANIFEST_LOC: request[PayloadParser.REQ_DROPOFF_PT], 
            PayloadParser.MANIFEST_SCHED_TIME: -1, 
            PayloadParser.MANIFEST_AMBULATORY: request[PayloadParser.REQ_AMBULATORY], 
            PayloadParser.MANIFEST_WHEELCHAIR: request[PayloadParser.REQ_WHEELCHAIR], 
            PayloadParser.MANIFEST_TIME_WINDOW_START: request[PayloadParser.REQ_DROPOFF_WINDOW_START],
            PayloadParser.MANIFEST_TIME_WINDOW_END: request[PayloadParser.REQ_DROPOFF_WINDOW_END]}
        
        # insert node ids for pickup and dropoff stops
        pickup_loc = pickup_stop[PayloadParser.MANIFEST_LOC]
        pickup_node_id = NetworkHandler.get_next_node_id(pickup_loc["lat"],pickup_loc["lon"])
        pickup_loc["node_id"] = pickup_node_id
        dropoff_loc = dropoff_stop[PayloadParser.MANIFEST_LOC]
        dropoff_node_id = NetworkHandler.get_next_node_id(dropoff_loc["lat"],dropoff_loc["lon"])
        dropoff_loc["node_id"] = dropoff_node_id

        load = 0
        state = driver_run_c[PayloadParser.DRIVER_STATE]
        pickup_stop[PayloadParser.MANIFEST_RUN_ID] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        dropoff_stop[PayloadParser.MANIFEST_RUN_ID] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        manifest = driver_run_c[PayloadParser.DRIVER_MANIFEST]
        state_loc = state[PayloadParser.DRIVER_STATE_LOC]
        node_id = NetworkHandler.get_next_node_id(state_loc["lat"],state_loc["lon"])
        state_loc["node_id"] = node_id
        start_node = Node(state_loc["lat"], state_loc["lon"], id =node_id)
        start_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
        completed_stops = []
        remaining_stops = []
        for stop in manifest:
            if stop[PayloadParser.MANIFEST_ORDER] <= state[PayloadParser.DRIVER_STATE_LOC_SERV]:
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    load += stop[PayloadParser.MANIFEST_AMBULATORY]
                else:
                    load -= stop[PayloadParser.MANIFEST_AMBULATORY]
                completed_stops.append(stop)
            else:
                remaining_stops.append(stop)
                stop_loc = stop[PayloadParser.MANIFEST_LOC]
                node_id = NetworkHandler.get_next_node_id(stop_loc["lat"], stop_loc["lon"])
                stop_loc["node_id"] = node_id
        
        NetworkHandler.initialize_travel_time_matrix()

        prev_cost = 0
        current_node = start_node
        for stop in remaining_stops:
            stop_loc = stop[PayloadParser.MANIFEST_LOC]
            next_node = Node(stop_loc["lat"],
                             stop_loc["lon"],
                             id=stop_loc["node_id"])
            prev_cost += NetworkHandler.travel_time(current_node,next_node)
            current_node = next_node

        end_time = state[PayloadParser.DRIVER_STATE_END_TIME]
        st_th = time.time()

        pool = Pool(processes=max(1,min(len(remaining_stops), 8)))
        args_list = [(i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective, depot_node, end_time, self.config.DWELL_PICKUP,
                    self.config.DWELL_ALIGHT) 
                    for i in range(len(remaining_stops) + 1) 
                    for j in range(i + 1, len(remaining_stops) + 2)]
        results = pool.map(OnlineRTVSolver._evaluate_insertion, args_list)
        pool.close()
        pool.join()

        best_cost = float("inf")
        best_insertion = None
        for cost, new_manifest in results:
            if cost < best_cost:
                best_cost = cost
                best_insertion = new_manifest

        if best_insertion is None:
            return -1,None

        new_driver_run = copy.deepcopy(driver_run)
        new_driver_run[PayloadParser.DRIVER_MANIFEST] = completed_stops + best_insertion
        new_driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_T_LOCS] = len(new_driver_run[PayloadParser.DRIVER_MANIFEST])
        if objective == "pick_up_time":
            return best_cost, new_driver_run
        return best_cost-prev_cost, new_driver_run

    def serve_asap(self, payload):
        """
        serve a request as soon as possible
        
        DO NOT USE IT for RTV generation process or COAML pipeline
        """
        # TODO check when this is valuable
        unserved_requests = []
        updated_driver_runs = copy.deepcopy(payload[PayloadParser.DRIVERS])
        for request in payload[PayloadParser.REQUESTS]:
            earliest_vehicle = None
            earliest_time = float("inf")
            earliest_vehicle_index = -1
            for vehicle_index in range(len(updated_driver_runs)):
                driver_run = updated_driver_runs[vehicle_index]
                pick_up_time, new_driver_run = self._insert_request_to_driver_run(payload[PayloadParser.DEPOT], driver_run, request, objective = "pick_up_time")
                if pick_up_time >=0 and pick_up_time < earliest_time:
                    earliest_time = pick_up_time
                    earliest_vehicle = new_driver_run
                    earliest_vehicle_index = vehicle_index
            if earliest_vehicle_index == -1:
                unserved_requests.append(request[PayloadParser.REQ_BOOKING_ID])
            else:
                updated_driver_runs[earliest_vehicle_index] = earliest_vehicle
        return updated_driver_runs, unserved_requests
    
    @staticmethod
    def finalize_driverRuns(config: Config, driver_runs: dict, depot_dict: dict) -> dict:
        """
        if config.return_depot finalize the vehicles by adding a depot stop to each ride, otherwise return input

        Behaviour: Depot_returns will be added straight from the last position of the vehicle where it finalized a previous trip, i.e., some vehicles might return already early during the day back to the depot despite more requests coming in. Our offline approach however has assigned all requests and thus, it is already fixed that no further requests have been accepted. 
        Alternative behaviour: Get back to the depot right before the final_end_time of each driver-run (condition depot_feasible confirms the options), but then depot_arrival_time would just be driver_run.state.end_time
        """
        if not config.RETURN_DEPOT:
            return driver_runs
        else: 
            driver_runs_c = copy.deepcopy(driver_runs) # keep oly one as is to not change information in place
            updated_driver_runs = []
            for run in driver_runs_c:
                driver_run = DriverRun.from_dict(run)
                if driver_run.manifest: # # manifest is not empty
                    # TODO check if those numbers are right
                    # Location are same, manifest seems to be more correct though
                    last_node = driver_run.state.loc
                    time_at_last_node = driver_run.state.location_dt_seconds # TODO why is this time_at_last_node different and this looks like an issue in the simulation?
                    # get last position and time from manifest
                    last_entry = driver_run.manifest[-1]
                    manifest_time = last_entry.scheduled_time
                    manifest_location = last_entry.loc
                    manifest_action = last_entry.action
                    assert manifest_action == VehicleStop.ACT_DROPOFF, f"Last stop {manifest_action} should have been a dropoff"

                    # TODO remove dwell time for ACT_DEPOT wherever that is
                    depot_node = Node.from_dict(depot_dict[PayloadParser.DEPOT_PT])
                    depot_arrival_time = time_at_last_node + NetworkHandler.travel_time(last_node, depot_node)
                    artificial_request_id = -(driver_run.state.run_id + 1)

                    depot_stop = ManifestEntry.from_dict({
                            PayloadParser.MANIFEST_RUN_ID: driver_run.state.run_id, 
                            PayloadParser.MANIFEST_BOOKING_ID: artificial_request_id, # easy recognition in the manifest
                            PayloadParser.MANIFEST_ORDER: driver_run.state.locations_already_serviced + 1, 
                            PayloadParser.MANIFEST_ACTION: VehicleStop.ACT_DEPOT, 
                            PayloadParser.MANIFEST_LOC: Node.to_dict(depot_node),
                            PayloadParser.MANIFEST_SCHED_TIME: depot_arrival_time,
                            PayloadParser.MANIFEST_AMBULATORY: 0, 
                            PayloadParser.MANIFEST_WHEELCHAIR: 0, 
                            PayloadParser.MANIFEST_TIME_WINDOW_START: depot_arrival_time-10, 
                            PayloadParser.MANIFEST_TIME_WINDOW_END: depot_arrival_time+10
                            })
                    driver_run.state.total_locations += 1
                    driver_run.state.locations_already_serviced += 1
                    driver_run.manifest.append(depot_stop)

                updated_driver_runs.append(driver_run.to_dict())
            OnlineRTVSolver._check_consistency_of_manifests(driver_runs_c, updated_driver_runs, [], [], keep_active=config.KEEP_ACTIVE, return_depot=config.RETURN_DEPOT, check_depot=True)
            
            return updated_driver_runs  