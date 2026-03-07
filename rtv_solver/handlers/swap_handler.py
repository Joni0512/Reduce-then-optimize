import numpy as np
import multiprocessing as mp
import gurobipy as gp
import time
import copy
import traceback

from gurobipy import GRB

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser

from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.config import Config

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class SwapHandler:
    """TODO docstring and behavior of this class"""
    def __init__(self, server_url, driver_runs, depot, config: Config):
        self.config = config
        self.MAX_NUM_THREAD = config.MAX_THREAD_CNT
        self.DWELL_PICKUP = config.DWELL_PICKUP
        self.DWELL_ALIGHT = config.DWELL_ALIGHT
        NetworkHandler.init_from_source(server_url)
        payload_object = PayloadParser.get_payload_object({
            PayloadParser.DRIVERS: driver_runs,
            PayloadParser.DEPOT: depot, 
            PayloadParser.REQUESTS: []})
        self.active_requests = set(payload_object.active_requests_keys)
        
        requests = payload_object.requests
        
        self.depot = payload_object.depot
        depot_node_id = NetworkHandler.get_next_node_id(self.depot.lat, self.depot.lon)
        self.depot.id = depot_node_id

        # add node_id to request pickup and dropoff spots
        for request in requests:
            # TODO this code can be moved to a helper function but it needs to change the information in-place
            req_pickup_pt, req_dropoff_pt = request[PayloadParser.REQ_PICKUP_PT], request[PayloadParser.REQ_DROPOFF_PT]
            node_id = NetworkHandler.get_next_node_id(req_pickup_pt["lat"],
                                                      req_pickup_pt["lon"])
            req_pickup_pt["node_id"] = node_id
            node_id = NetworkHandler.get_next_node_id(req_dropoff_pt["lat"], req_dropoff_pt["lon"])
            req_dropoff_pt["node_id"] = node_id
        
        self.request_dic = {}
        for request in requests:
            self.request_dic[request[PayloadParser.REQ_BOOKING_ID]] = request

        self.driver_runs = copy.deepcopy(driver_runs)
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            state_loc = state[PayloadParser.DRIVER_STATE_LOC]
            current_node_id = NetworkHandler.get_next_node_id(state_loc["lat"], 
                                                              state_loc["lon"])
            state_loc["node_id"] = current_node_id
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            for stop in manifest[current_order:]:
                booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                stop_loc = stop[PayloadParser.MANIFEST_LOC]
                request = self.request_dic[booking_id]
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    stop_loc["node_id"] = request[PayloadParser.REQ_PICKUP_PT]["node_id"]
                else:
                    stop_loc["node_id"] = request[PayloadParser.REQ_DROPOFF_PT]["node_id"]
            
        # Initialize travel time matrix
        NetworkHandler.initialize_travel_time_matrix()

    def run_swap(self, rerunning=False):
        console_logger.debug("Started swap round")
        if rerunning:
            self.driver_runs = copy.deepcopy(self.new_driver_runs)
        # collect requests per vehicle that have not been served yet
        driver_run_requests = {}
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            driver_run_requests[run_id] = set()
            for stop in manifest[current_order:]:
                booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    driver_run_requests[run_id].add(booking_id)
        
        SwapHandler.manifest_options = []
        initial_cost = 0
        pool = mp.Pool(self.MAX_NUM_THREAD)
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            active_requests_in_manifest = driver_run_requests[run_id]
            manifest_cost = SwapHandler._manifest_cost(driver_run)
            initial_cost += manifest_cost
            SwapHandler.manifest_options.append((run_id,active_requests_in_manifest,manifest_cost,driver_run,0))

            for other_booking_id in self.active_requests:
                if other_booking_id in active_requests_in_manifest:
                    continue
                request = self.request_dic[other_booking_id]
                requests_after = active_requests_in_manifest.copy()
                requests_after.add(other_booking_id)

                args = (self.depot, run_id, requests_after, driver_run, request, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                pool.apply_async(SwapHandler.create_manifest_option, 
                                 args=args, 
                                 callback=SwapHandler._process_swap_result,
                                 error_callback=self._on_worker_error)

            for booking_id in active_requests_in_manifest:
                driver_run_without_request = SwapHandler._remove_request_from_driver_run(driver_run, booking_id, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                cost = SwapHandler._manifest_cost(driver_run_without_request)
                requests_in_new_manifest = active_requests_in_manifest.copy()
                requests_in_new_manifest.remove(booking_id)
                SwapHandler.manifest_options.append((run_id, requests_in_new_manifest, cost, driver_run_without_request,0))

                for other_booking_id in self.active_requests:
                    if other_booking_id in active_requests_in_manifest:
                        continue
                    request = self.request_dic[other_booking_id]
                    requests_after = requests_in_new_manifest.copy()
                    requests_after.add(other_booking_id)

                    args = (self.depot, run_id, requests_after, driver_run_without_request, request, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                    pool.apply_async(SwapHandler.create_manifest_option, 
                                     args=args, 
                                     callback=SwapHandler._process_swap_result,
                                     error_callback=self._on_worker_error)

                    # cost, driver_run_with_new_request = SwapHandler.insert_request_to_driver_run(
                    #     self.depot, driver_run_without_request, request, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                    # SwapHandler.manifest_options.append((run_id, requests_after, cost, driver_run_with_new_request))
        pool.close()
        pool.join()

        # Filter out invalid manifest options
        SwapHandler.infeasible_manifest_options = [option for option in SwapHandler.manifest_options if option[2] == -1]
        SwapHandler.manifest_options = [option for option in SwapHandler.manifest_options if option[2] != -1]

        no_options = len(SwapHandler.manifest_options)
        manifests_with_request = {}
        for booking_id in self.active_requests:
            manifests_with_request[booking_id] = []
            for i in range(no_options):
                option = SwapHandler.manifest_options[i]
                if booking_id in option[1]:
                    manifests_with_request[booking_id].append(i)
        
        manifests_with_vehicle = {}
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            manifests_with_vehicle[run_id] = []
            for i in range(no_options):
                option = SwapHandler.manifest_options[i]
                if option[0] == run_id:
                    manifests_with_vehicle[run_id].append(i)

        selected_options = []
        console_logger.debug("Number of manifest options: {0}".format(no_options))
        console_logger.debug("Started building optimization problem")
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model('Swap assignment',env=env)
            var_type = GRB.BINARY
            trip_costs = np.zeros(no_options)
            for i in range(no_options):
                trip_costs[i] = SwapHandler.manifest_options[i][2]
            x_t = m.addVars(no_options, lb=0, ub=1, obj=trip_costs, name="t", vtype=var_type)

            m.addConstrs((gp.quicksum(x_t[i] for i in manifests_with_vehicle[run_id]) == 1 for run_id in list(manifests_with_vehicle.keys())), "driver_runs")

            m.addConstrs((gp.quicksum(x_t[i] for i in manifests_with_request[booking_id]) == 1 for booking_id in list(manifests_with_request.keys())), "requests")

            m.setParam('TimeLimit', 10)
            m.optimize()

            if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
                console_logger.info("SwapILP time: {0}s".format(m.Runtime))

                for i in range(no_options):
                    if x_t[i].X == 1:
                        selected_options.append(SwapHandler.manifest_options[i])
            else:
                m.Params.OutputFlag = 1
                m.computeIIS()
                m.write("infeasible.ilp")   # human-readable
                m.write("infeasible.lp")    # full model
                m.write("infeasible.mps")   # optional

                # Print which constraints are in IIS
                console_logger.error("\n--- IIS constraints ---")
                for constraint in m.getConstrs():
                    if constraint.IISConstr:
                        console_logger.error("IIS:", constraint.ConstrName)
                raise Exception("Gurobi solver ended with code: {0}".format(m.Status))
        
        new_cost = 0
        no_of_swaps = 0
        selected_driver_runs = {}
        for run_id, active_requests, cost, driver_run, time_taken in selected_options:
            new_cost += cost
            selected_driver_runs[run_id] = driver_run
            prev_requests_in_manifest = driver_run_requests[run_id]
            uncommon_items = prev_requests_in_manifest.symmetric_difference(active_requests)
            num_uncommon = len(uncommon_items)
            no_of_swaps += num_uncommon
        no_of_swaps //= 2  # Each swap is counted twice (once for each driver run)
        console_logger.info('Initial cost: {0}, new cost: {1}, cost reduction'.format(initial_cost, new_cost, initial_cost-new_cost))
        console_logger.info('Number of swaps: {0}'.format(no_of_swaps))

        self.new_driver_runs = []
        run_ids = list(selected_driver_runs.keys())
        run_ids.sort()
        for run_id in run_ids:
            driver_run = selected_driver_runs[run_id]
            self.new_driver_runs.append(driver_run)

        return self.new_driver_runs, initial_cost-new_cost, no_of_swaps

    @staticmethod
    def create_manifest_option(depot_node, run_id, requests, driver_run, request, DWELL_PICKUP, DWELL_ALIGHT):
        cost, new_driver_run, time_taken = SwapHandler._insert_request_to_driver_run(
            depot_node, 
            driver_run, 
            request, 
            DWELL_PICKUP, 
            DWELL_ALIGHT)
        return (run_id, requests, cost, new_driver_run, time_taken)

    @staticmethod
    def _process_swap_result(result):
        SwapHandler.manifest_options.append(result)

    @staticmethod
    def _on_worker_error(e):
        console_logger.error("Worker crashed:", repr(e))
        traceback.print_exc()
        raise e
    
    @staticmethod
    def _manifest_cost(driver_run):
        cost = 0

        state = driver_run[PayloadParser.DRIVER_STATE]
        no_completed_stops = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        remaining_stops = driver_run[PayloadParser.DRIVER_MANIFEST][no_completed_stops:]

        current_node = SwapHandler._stop_to_node(state)
        for stop in remaining_stops:
            next_node = SwapHandler._stop_to_node(stop)
            cost += NetworkHandler.travel_time(current_node,next_node)
            current_node = next_node
        return cost

    @staticmethod
    def _stop_to_node(stop):
        stop_loc = stop[PayloadParser.MANIFEST_LOC]
        return Node(stop_loc["lat"],
                    stop_loc["lon"],
                    id = stop_loc["node_id"])

    @staticmethod
    def _remove_request_from_driver_run(driver_run, booking_id, DWELL_PICKUP, DWELL_ALIGHT):
        state = copy.deepcopy(driver_run[PayloadParser.DRIVER_STATE])
        no_completed_stops = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        menifest = copy.deepcopy(driver_run[PayloadParser.DRIVER_MANIFEST])
        new_manifest = menifest[:no_completed_stops]
        remaining_stops = []

        current_node = SwapHandler._stop_to_node(state)
        current_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
        current_order = no_completed_stops
        for stop in menifest[no_completed_stops:]:
            if stop[PayloadParser.MANIFEST_BOOKING_ID] == booking_id:
                continue
            stop_node = SwapHandler._stop_to_node(stop)
            travel_time = NetworkHandler.travel_time(current_node, stop_node)
            current_time += travel_time
            if current_time < stop[PayloadParser.MANIFEST_TIME_WINDOW_START]:
                current_time = stop[PayloadParser.MANIFEST_TIME_WINDOW_START]
            stop[PayloadParser.MANIFEST_SCHED_TIME] = current_time
            current_order += 1
            stop[PayloadParser.MANIFEST_ORDER] = current_order

            if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                current_time += DWELL_PICKUP
            else:
                current_time += DWELL_ALIGHT

            remaining_stops.append(stop)
            current_node = stop_node

        state[PayloadParser.DRIVER_STATE_T_LOCS] = len(new_manifest + remaining_stops)
        
        return {
            PayloadParser.DRIVER_STATE: state,
            PayloadParser.DRIVER_MANIFEST: new_manifest + remaining_stops
        }

    @staticmethod  
    def _insert_request_to_driver_run(depot_node, driver_run, request, DWELL_PICKUP, DWELL_ALIGHT) -> tuple[float, ]:
        driver_run_c = copy.deepcopy(driver_run)
        # initialize stops
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

        load = 0
        state = driver_run_c[PayloadParser.DRIVER_STATE]
        manifest = driver_run_c[PayloadParser.DRIVER_MANIFEST]
        # insert vehicle to stops
        pickup_stop[PayloadParser.MANIFEST_RUN_ID] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        dropoff_stop[PayloadParser.MANIFEST_RUN_ID] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        
        start_node = SwapHandler._stop_to_node(state)
        start_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
        # differentiate already completed stops and remaining stops in manifest
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

        # get pickup and dropoff indices in the manifest
        earliest_pickup_time = pickup_stop[PayloadParser.MANIFEST_TIME_WINDOW_START]
        latest_pickup_time = pickup_stop[PayloadParser.MANIFEST_TIME_WINDOW_END]
        earliest_dropoff_time = dropoff_stop[PayloadParser.MANIFEST_TIME_WINDOW_START]
        latest_dropoff_time = dropoff_stop[PayloadParser.MANIFEST_TIME_WINDOW_END]

        pick_earliest_index = 0
        pick_latest_index = 0
        for i, stop in enumerate(remaining_stops):
            if stop[PayloadParser.MANIFEST_TIME_WINDOW_END] >= earliest_pickup_time:
                break
            else:
                pick_earliest_index = i + 1
        
        for i, stop in enumerate(remaining_stops):
            if stop[PayloadParser.MANIFEST_TIME_WINDOW_START] > latest_pickup_time:
                break
            else:
                pick_latest_index = i + 1
        if pick_latest_index == len(remaining_stops):
            pick_latest_index += 1

        drop_earliest_index = 0
        drop_latest_index = 0
        for i, stop in enumerate(remaining_stops):
            if stop[PayloadParser.MANIFEST_TIME_WINDOW_END] >= earliest_dropoff_time:
                break
            else:
                drop_earliest_index = i + 1
        for i, stop in enumerate(remaining_stops):
            if stop[PayloadParser.MANIFEST_TIME_WINDOW_START] > latest_dropoff_time:
                break
            else:
                drop_latest_index = i + 1
        if drop_latest_index == len(remaining_stops):
            drop_latest_index += 1

        st_time = time.time()
        # build args for insertion evaluation
        end_time = state[PayloadParser.DRIVER_STATE_END_TIME]
        objective = "vmt"
        args_list = [ # create args for each pickup index i & each dropoff index
            (i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective,depot_node, end_time, DWELL_PICKUP, DWELL_ALIGHT) 
            for i in range(pick_earliest_index, pick_latest_index) # all possible indices for pickup
            for j in range(max(i, drop_earliest_index) + 1, min(len(remaining_stops) +1, drop_latest_index) + 1)]
        
        # TODO why is this not parallelized?
        results = [SwapHandler._evaluate_insertion(args) for args in args_list]

        time_taken_to_evaluate = time.time() - st_time # why do we return the time_taken and never use it

        best_cost = float("inf")
        best_insertion = None
        for cost, new_manifest in results:
            if cost < best_cost:
                best_cost = cost
                best_insertion = new_manifest

        if best_insertion is None:
            return -1, None, time_taken_to_evaluate

        new_driver_run = copy.deepcopy(driver_run)
        new_driver_run[PayloadParser.DRIVER_MANIFEST] = completed_stops + best_insertion
        new_driver_run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_T_LOCS] = len(new_driver_run[PayloadParser.DRIVER_MANIFEST])

        return best_cost, new_driver_run, time_taken_to_evaluate

    @staticmethod
    def _evaluate_insertion(args) -> tuple[float, dict]:
        i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective, depot, end_time, dwell_pickup, dwell_alight = args # unpack args
        # check all different insertions where one trip (pickup + dropoff is added to an existing manifest)
        new_manifest = copy.deepcopy(remaining_stops[:i] + [pickup_stop] + remaining_stops[i:j] + [dropoff_stop] + remaining_stops[j:])
        current_time = start_time
        current_node = start_node
        current_load = load
        # initialize loop
        cost = 0
        order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        index = 0
        for stop in new_manifest:
            next_node = SwapHandler._stop_to_node(stop)
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
            # increment elements
            order += 1
            stop[PayloadParser.MANIFEST_ORDER] = order
            index += 1

        if current_time + NetworkHandler.travel_time(current_node,depot) > end_time:
            return float("inf"), None
        if objective == "pick_up_time": # result would not be cost but time?
            return new_manifest[i][PayloadParser.MANIFEST_SCHED_TIME], new_manifest
        return cost, new_manifest
