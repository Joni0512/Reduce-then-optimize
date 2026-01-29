from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler, VehicleHandlerConfig
from rtv_solver.handlers.trip_handler import TripHandler, TripHandlerConfig
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.swap_handler import SwapHandler
from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop
import copy
import multiprocessing
import sys
from multiprocessing import Pool
import time
import numpy as np
import logging
import copy

class OnlineRTVSolver:
    """solves entire RTV problem for a given payload """
    def __init__(self, config = None):
        self.config = config
        self.SERVER_URL = self.config.server_url
        self.MAX_CARDINALITY = self.config.max_cardinality
        self.MAX_THREAD_CNT = self.config.max_thread_cnt
        self.DWELL_PICKUP = self.config.dwell_pickup
        self.DWELL_ALIGHT = self.config.dwell_alight

        # NOTE this still seems like a code smell, but makes it a bit easier to read below
        # TODO self.SH_CONFIG = SwapHandlerConfig()
        self.TH_CONFIG = TripHandlerConfig(
                ilp_solver_timeout      = self.config.ilp_timeout,
                penalty                 = self.config.ilp_penalty,
                max_cardinality         = self.config.max_cardinality, 
                max_thread_cnt          = self.config.max_thread_cnt,
                shareable_cost_factor   = self.config.share_cost_factor,
                rebalancing             = self.config.rebalancing,
                rtv_timeout             = self.config.rtv_timeout,
            )
        
        self.VH_CONFIG = VehicleHandlerConfig(
                output_directory = self.config.output_dir,
                largest_tsp = self.config.largest_tsp
        )

        if sys.platform == "darwin": # required to run online_solver correctly on MacOS
            try:
                multiprocessing.set_start_method("fork")
            except RuntimeError: # start method was already set somewhere else -> don't crash
                pass

    def check_feasibility(self, payload):
        # NOTE for what do we need this method?
        NetworkHandler.init(True, self.SERVER_URL)
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
                cost, _ = self.insert_request_to_driver_run(
                    payload[PayloadParser.DEPOT], driver_run, request_copy)
                if cost >= 0 and cost < best_cost:
                    best_cost = cost
            if best_cost < float("inf"):
                feasible_time_slots.append((time_window,best_cost / request_travel_time))

        return feasible_time_slots

    def resolve_pdptw_rtv(self, payload):
        updated_driver_runs, unserved_requests = self.solve_pdptw_rtv(payload)
        if len(unserved_requests) == 0:
            return updated_driver_runs
        else:
            return payload

    def solve_pdptw_rtv(self, payload, iteration = 0):
        # initalize network and payload
        NetworkHandler.init(True, self.SERVER_URL)
        payload_object = PayloadParser.get_payload_object(payload)
        # get all requests of payload, add 
        request_handler = RequestHandler(payload_object.requests, self.DWELL_PICKUP, self.DWELL_ALIGHT)
        temp_batch = request_handler.get_all_requests()
        
        # filter active and boarded requests for subsequent action as they need to be integrated when handling new trip generation
        batch = []
        active_requests = {}
        boarded_requests = {}

        for req in temp_batch:
            req_id = req.id
            if req_id in payload_object.boarded_requests_keys:
                boarded_requests[req_id] = req
            else:
                if req_id in payload_object.active_requests_keys:
                    active_requests[req_id] = req
                batch.append(req)
        
        # initialize all vehicles as they are stores in the original payload-object
        vehicle_handler = VehicleHandler(payload_object.depot, 
                                         payload_object.driver_runs,
                                         self.VH_CONFIG)
        # create trips of all already boarded requests
        boarded_trips = TripHandler.create_trip_for_picked_requests(boarded_requests, iteration)
        # update vehicle position/trips/times along its path according to all data stored in the manifest
        vehicle_handler.add_manifest_to_vehicles(payload_object.driver_runs,
                                                 boarded_requests,
                                                 boarded_trips, 
                                                 self.DWELL_ALIGHT, 
                                                 self.DWELL_PICKUP)
        
        NetworkHandler.initialize_travel_time_matrix()
        iteration += 1  # increase iteration as the prior step was just rebuilding from the last iteration (if there was a prior step)
        unserved_requests = set([req.id for req in batch]) - set(active_requests.keys()) # number of requests that are not already confirmed to be served
        try:
            # generate and assign trips to each vehicle using the RTV approach solved by an ILP
            trip_handler = TripHandler(
                vehicle_handler.vehicles,
                batch, 
                active_requests, 
                iteration, 
                self.TH_CONFIG)
        except Exception as e:
            raise e
        
        # assign vehicles and add trips / sequence to each vehicle 
        vehicle_assignment = trip_handler.get_veh_assignment()
        for vehicle_id in vehicle_assignment:
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips, prev_sequence = vehicle_assignment[vehicle_id]
            plan = VehicleHandler.plan_trip_insertions(vehicle, trips, prev_sequence=prev_sequence)
            vehicle.apply_trip_insertion(plan)
            for trip in trips: # remove assigned trips from unserved
                if trip.request_id in unserved_requests:
                    unserved_requests.remove(trip.request_id)

        # update driver runs
        updated_driver_runs = []
        for driver_run in payload_object.driver_runs:
            new_driver_run = self.update_run(vehicle_handler, driver_run)
            updated_driver_runs.append(new_driver_run)
            
        # check invariants whether manifest is still correct
        self.check_consistency_of_manifests(payload[PayloadParser.DRIVERS], 
                                            updated_driver_runs,
                                            unserved_requests, 
                                            payload[PayloadParser.REQUESTS])
        return updated_driver_runs, list(unserved_requests) # ,trip_handler, vehicle_handler, request_handler, payload_object

    @staticmethod
    def check_consistency_of_manifests(prev_driver_runs: list[dict], new_driver_runs: list[dict], unserved_requests: set[int], new_requests: list[dict]):
        """ 
        check if each requests are picked up AND dropped off exactly once,
        unserved requests should not appear in the manifests
        """
        # initialize all new requests that need to be picked, dropped or unserved
        picked_requests = set([req["booking_id"] for req in new_requests])
        dropped_requests = copy.deepcopy(picked_requests) # set([req["booking_id"] for req in new_requests]) # simplified as it does not have to run the same loop twice
        # get all requests that are already in the previous manifest
        for driver_run in prev_driver_runs:
            for stop in driver_run[PayloadParser.DRIVER_MANIFEST]:
                # TODO why is type of stop[booking_id] a string and the set of request_ids floats (wouldn't int suffice?)
                # float is quickfix for type mismatch - id in stop is stored as string instead of float
                stop_id = float(stop[PayloadParser.MANIFEST_BOOKING_ID]) # prev: stop_id = stop["booking_id"]
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    picked_requests.add(stop_id)
                else:
                    dropped_requests.add(stop_id)
        # remove all requests that are picked up/dropped off
        for driver_run in new_driver_runs:
            for stop in driver_run[PayloadParser.DRIVER_MANIFEST]:
                stop_id = float(stop[PayloadParser.MANIFEST_BOOKING_ID]) 
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    picked_requests.remove(stop_id)
                else:
                    dropped_requests.remove(stop_id)
        # remove all requests that are unserved            
        for req_id in unserved_requests:
            req_id = float(req_id)
            picked_requests.remove(req_id)
            dropped_requests.remove(req_id)

        if len(picked_requests) > 0 or len(dropped_requests) > 0:
            # TODO: fails with wilson_data, cardinality = 3, thread_cnt = 16, batch_interval = 1800, step_size = 1800 (should be reproducible with this)
            print("Missing requests:", picked_requests, dropped_requests)
            raise Exception("Error: Some requests could not be removed.")
        return True

    @staticmethod
    def update_run(vehicle_handler: VehicleHandler, driver_run: dict) -> dict:
        """
        Update manifest of a driver_run by keeping all already-served stops and regenerating the remaining stops from the vehicle's stop_sequence.
        Returns a new driver_run dict (state + manifest).
        """
        # retrieve old information
        state = driver_run[PayloadParser.DRIVER_STATE]
        old_manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
        current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        
        vehicle_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
        vehicle = vehicle_handler.vehicles[vehicle_id]

        # Keep already served part, rebuild future part
        new_manifest = old_manifest[:current_order]
        added_manifest = vehicle_handler.get_manifest(vehicle, current_order)
        new_manifest.extend(added_manifest)
        # Update state meta info
        new_state = state.copy()
        new_state[PayloadParser.DRIVER_STATE_T_LOCS] = len(new_manifest)
        # Build new driver run from both parts
        new_driver_run = {PayloadParser.DRIVER_STATE: new_state, 
                              PayloadParser.DRIVER_MANIFEST: new_manifest}
        return new_driver_run

    
    def simulate_manifest(self, current_time, driver_runs, intermediate_location=True):
        """deprecated - used in main.py (disfunctional)"""
        NetworkHandler.init(True, self.SERVER_URL)
        new_driver_runs = []
        # TODO longterm: turn driver_run into an object that handles all the conditions and changes based on validated calls
        for driver_run in driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]

            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            next_immediate_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
            next_immediate_loc = state[PayloadParser.DRIVER_STATE_LOC]
            
            # update time if manifest is already completed
            if len(manifest) == current_order and next_immediate_time < current_time:
                next_immediate_time = current_time

            while len(manifest) > current_order and current_time >= manifest[current_order][PayloadParser.MANIFEST_SCHED_TIME]:
                next_stop = manifest[current_order]
                next_immediate_time = next_stop[PayloadParser.MANIFEST_SCHED_TIME]
                next_immediate_loc = next_stop[PayloadParser.MANIFEST_LOC]
                # apply dwell time if applicable
                if next_stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    next_immediate_time += self.DWELL_PICKUP
                else:
                    next_immediate_time += self.DWELL_ALIGHT
                current_order+=1
                if next_immediate_time > current_time:
                    break
                
            if len(manifest) > current_order and next_immediate_time < current_time and intermediate_location:
                next_immediate_node = NetworkHandler.get_node_from_manifest_location(next_immediate_loc)
                target_node = NetworkHandler.get_node_from_manifest_location(manifest[current_order][PayloadParser.MANIFEST_LOC])
                next_immediate_time, next_immediate_node = NetworkHandler.get_current_location_time(
                    next_immediate_node, target_node, next_immediate_time, current_time)
                next_immediate_loc = {"lat":next_immediate_node.lat,
                                      "lon":next_immediate_node.lon}

            state[PayloadParser.DRIVER_STATE_DT_SEC] = next_immediate_time
            state[PayloadParser.DRIVER_STATE_LOC] = next_immediate_loc
            state[PayloadParser.DRIVER_STATE_LOC_SERV] = current_order
            new_driver_runs.append({
                PayloadParser.DRIVER_STATE: state,
                PayloadParser.DRIVER_MANIFEST: manifest})
        
        self.check_consistency_of_manifests(driver_runs, new_driver_runs, [], [])
        return new_driver_runs

    def solve_pdptw_heuristic(self, payload, return_added_vmt=False):
        updated_driver_runs = copy.deepcopy(payload[PayloadParser.DRIVERS])
        total_cost = 0
        unserved_requests = []
        for request in payload[PayloadParser.REQUESTS]:
            cheapest_vehicle = None
            cheapest_cost = float("inf")
            cheapest_vehicle_index = -1
            for vehicle_index in range(len(updated_driver_runs)):
                driver_run = updated_driver_runs[vehicle_index]
                cost, new_driver_run = self.insert_request_to_driver_run(payload[PayloadParser.DEPOT], driver_run, request)
                if cost >=0 and cost < cheapest_cost:
                    cheapest_cost = cost
                    cheapest_vehicle = new_driver_run
                    cheapest_vehicle_index = vehicle_index
            if cheapest_vehicle is not None:
                updated_driver_runs[cheapest_vehicle_index] = cheapest_vehicle
                total_cost += cheapest_cost
            else:
                unserved_requests.append(request[PayloadParser.REQ_BOOKING_ID])
        
        self.check_consistency_of_manifests(payload[PayloadParser.DRIVERS], updated_driver_runs, unserved_requests, payload[PayloadParser.REQUESTS])
        if return_added_vmt:
            return updated_driver_runs, unserved_requests, total_cost
        return updated_driver_runs, unserved_requests

    def solve_pdptw(self, payload, skip_swapping=True):
        # NOTE what is the difference to PDPTW_RTV
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
        if remaining_requests.max() <= self.MAX_CARDINALITY:
            updated_driver_runs, unserved_requests = self.solve_pdptw_rtv(payload)
            if len(unserved_requests) == 0:
                return updated_driver_runs, unserved_requests

        # Use heuristic if any vehicle has too many remaining requests
        logging.debug("Inserting with heuristic...")
        # Get the initial solution with insertion heuristic
        updated_driver_runs, unserved_requests = self.solve_pdptw_heuristic(payload)
        if len(unserved_requests) > 0:
            logging.debug("Unserved requests after heuristic: %d", len(unserved_requests))
            # Return without further optimization if there are unserved requests
            return updated_driver_runs, unserved_requests

        if skip_swapping:
            return updated_driver_runs, unserved_requests
        # If all requests are served, try to optimize the solution further
        logging.debug("Optimizing solution with swap heuristic...")
        start_time = time.time()
        swap_handler = SwapHandler(self.SERVER_URL,
                                   updated_driver_runs,
                                   payload[PayloadParser.DEPOT],
                                   self.DWELL_PICKUP, 
                                   self.DWELL_ALIGHT, 
                                   self.MAX_THREAD_CNT)
        swaped_driver_runs, reduced_cost, no_of_swaps = swap_handler.run_swap()
        while no_of_swaps > 0 and reduced_cost > 0 and time.time() - start_time < self.RTV_TIMEOUT:
            updated_driver_runs = swaped_driver_runs
            swaped_driver_runs, reduced_cost, no_of_swaps = swap_handler.run_swap(rerunning=True)

        return swaped_driver_runs, unserved_requests

    def evaluate_insertion(args):
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
                             identifier = stop_location["node_id"])
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

    def insert_request_to_driver_run(self, depot, driver_run, request, objective="vmt"):
        NetworkHandler.init(True, self.SERVER_URL)
        driver_run_c = copy.deepcopy(driver_run)
        depot_pt = depot[PayloadParser.DEPOT_PT]
        depot_node_id = NetworkHandler.get_next_node_id(depot_pt["lat"], depot_pt["lon"])
        depot_node = Node(
            depot_pt["lat"], 
            depot_pt["lon"], 
            identifier=depot_node_id)

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
        state = driver_run_c[PayloadParser.DRIVER_STATE] # TODO JW: do we want to have a copy here? or what does the 'c' stand for?
        pickup_stop[PayloadParser.MANIFEST_RUN_ID] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        dropoff_stop[PayloadParser.MANIFEST_RUN_ID] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        manifest = driver_run_c[PayloadParser.DRIVER_MANIFEST]
        state_loc = state[PayloadParser.DRIVER_STATE_LOC]
        node_id = NetworkHandler.get_next_node_id(state_loc["lat"],state_loc["lon"])
        state_loc["node_id"] = node_id
        start_node = Node(state_loc["lat"], state_loc["lon"], identifier=node_id)
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
                             identifier=stop_loc["node_id"])
            prev_cost += NetworkHandler.travel_time(current_node,next_node)
            current_node = next_node

        end_time = state[PayloadParser.DRIVER_STATE_END_TIME]
        st_th = time.time()

        pool = Pool(processes=max(1,min(len(remaining_stops), 8)))
        args_list = [(i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective, depot_node, end_time, self.DWELL_PICKUP,
                    self.DWELL_ALIGHT) 
                    for i in range(len(remaining_stops) + 1) 
                    for j in range(i + 1, len(remaining_stops) + 2)]
        results = pool.map(OnlineRTVSolver.evaluate_insertion, args_list)
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
        unserved_requests = []
        updated_driver_runs = copy.deepcopy(payload[PayloadParser.DRIVERS])
        for request in payload[PayloadParser.REQUESTS]:
            earliest_vehicle = None
            earliest_time = float("inf")
            earliest_vehicle_index = -1
            for vehicle_index in range(len(updated_driver_runs)):
                driver_run = updated_driver_runs[vehicle_index]
                pick_up_time, new_driver_run = self.insert_request_to_driver_run(payload[PayloadParser.DEPOT], driver_run, request, objective = "pick_up_time")
                if pick_up_time >=0 and pick_up_time < earliest_time:
                    earliest_time = pick_up_time
                    earliest_vehicle = new_driver_run
                    earliest_vehicle_index = vehicle_index
            if earliest_vehicle_index == -1:
                unserved_requests.append(request[PayloadParser.REQ_BOOKING_ID])
            else:
                updated_driver_runs[earliest_vehicle_index] = earliest_vehicle
        return updated_driver_runs, unserved_requests

    def get_stats(self, depot, driver_runs, travel_time_error_margin=5):
        """
        deprecated with new StatsParser class
        """
        feasible = True
        stats = {
            "vmt": 0,
            "pmt": 0,
            "vmt/pmt": 0,
            "serviced": 0,
            "average_wait_time": 0,
            "average_detour": 0,
            "wait_time": [],
            "detour": [],
        }

        NetworkHandler.init(True, self.SERVER_URL)
        request_stops = {}
        for driver_run in driver_runs:
            load = 0
            current_node = Node(depot[PayloadParser.DEPOT_PT]["lat"], 
                                depot[PayloadParser.DEPOT_PT]["lon"])
            current_time = driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_START_TIME]
            for stop in driver_run[PayloadParser.DRIVER_MANIFEST]:
                booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                if booking_id not in request_stops:
                    request_stops[booking_id] = {}
                action = stop[PayloadParser.MANIFEST_ACTION]
                served_time = stop[PayloadParser.MANIFEST_SCHED_TIME]
                next_node = Node(stop["loc"]["lat"],stop["loc"]["lon"])
                duration = NetworkHandler.travel_time(current_node,next_node)
                stats["vmt"] += duration
                current_time += duration
                if current_time > served_time + travel_time_error_margin:  # Allow a small margin of error
                    feasible = False
                    print("Error: Scheduled time is impossible ", current_time-served_time)
                    if duration > 0:
                        print(100*(current_time - served_time)/duration)
                    print("Current time: ",current_time)
                    print("Scheduled time: ",served_time)
                    print(stop)
                if current_time < served_time:
                    current_time = served_time
                
                if served_time < stop[PayloadParser.MANIFEST_TIME_WINDOW_START]:
                    feasible = False
                    print("Error: Served before window start")
                if served_time > stop[PayloadParser.MANIFEST_TIME_WINDOW_END]:
                    feasible = False
                    print("Error: Served after window end")
                if action == VehicleStop.ACT_PICKUP:
                    load += stop[PayloadParser.MANIFEST_AMBULATORY]
                    current_time += 180
                    if "pick_up" in request_stops[booking_id]:
                        print("Error: Pick up already exists")
                    request_stops[booking_id]["pick_up"] = stop
                else:
                    current_time += 60
                    load -= stop[PayloadParser.MANIFEST_AMBULATORY]
                    if "drop_off" in request_stops[booking_id]:
                        feasible = False
                        print("Error: Drop off already exists")
                    if "pick_up" not in request_stops[booking_id]:
                        feasible = False
                        print("Error: Drop off before pick up")
                    request_stops[booking_id]["drop_off"] = stop
                    stats["serviced"] += 1
                if load > driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_AM_CAP]:
                    feasible = False
                    print("Error: Over capacity")
                current_node = next_node

        for served in request_stops:
            if "drop_off" not in request_stops[served]:
                feasible = False
                print("Error: Request not dropped off")
            origin = Node(request_stops[served]["pick_up"]["loc"]["lat"],request_stops[served]["pick_up"]["loc"]["lon"])
            destination = Node(request_stops[served]["drop_off"]["loc"]["lat"],request_stops[served]["drop_off"]["loc"]["lon"])
            travel_time = NetworkHandler.travel_time(origin,destination)
            stats["pmt"] += travel_time
            stats["wait_time"].append(
                request_stops[served]["pick_up"]["scheduled_time"]-
                request_stops[served]["pick_up"]["time_window_start"])
            stats["detour"].append(request_stops[served]["drop_off"]["scheduled_time"]-
                                   request_stops[served]["pick_up"]["scheduled_time"]-
                                   travel_time)
        if stats["pmt"] > 0:
            stats["vmt/pmt"] = stats["vmt"] / stats["pmt"]
        if stats["serviced"] > 0:
            stats["average_wait_time"] = sum(stats["wait_time"]) / stats["serviced"]
            stats["average_detour"] = sum(stats["detour"]) / stats["serviced"]
        return feasible, stats