from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.node import Node
import numpy as np
import logging
import multiprocessing as mp
import gurobipy as gp
from gurobipy import GRB
import time
import copy

class SwapHandler:
    def __init__(self, server_url, driver_runs, depot, DWELL_PICKUP, DWELL_ALIGHT, MAX_NUM_THREAD):
        self.MAX_NUM_THREAD = MAX_NUM_THREAD
        NetworkHandler.init(True, server_url)
        payload_object = PayloadParser.get_payload_object({"driver_runs": driver_runs, "depot": depot, "requests": []})
        self.active_requests = set(payload_object.active_requests)
        requests = payload_object.requests

        depot_node_id = NetworkHandler.get_next_node_id(payload_object.depot.lat, payload_object.depot.lon)
        payload_object.depot.id = depot_node_id
        for request in requests:
            node_id = NetworkHandler.get_next_node_id(request["pickup_pt"]["lat"], request["pickup_pt"]["lon"])
            request["pickup_pt"]["node_id"] = node_id
            node_id = NetworkHandler.get_next_node_id(request["dropoff_pt"]["lat"], request["dropoff_pt"]["lon"])
            request["dropoff_pt"]["node_id"] = node_id
        
        self.request_dic = {}
        for request in requests:
            self.request_dic[request["booking_id"]] = request

        self.driver_runs = copy.deepcopy(driver_runs)
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            current_node_id = NetworkHandler.get_next_node_id(state[PayloadParser.DRIVER_STATE_LOC]["lat"], state[PayloadParser.DRIVER_STATE_LOC]["lon"])
            state[PayloadParser.DRIVER_STATE_LOC]["node_id"] = current_node_id
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            for stop in manifest[current_order:]:
                booking_id = stop["booking_id"]
                request = self.request_dic[booking_id]
                if stop["action"] == "pickup":
                    stop["loc"]["node_id"] = request["pickup_pt"]["node_id"]
                else:
                    stop["loc"]["node_id"] = request["dropoff_pt"]["node_id"]
            
        # Initialize the travel time matrix
        NetworkHandler.initialize_travel_time_matrix()

        self.DWELL_PICKUP = DWELL_PICKUP
        self.DWELL_ALIGHT = DWELL_ALIGHT
        self.depot = payload_object.depot

    def run_swap(self):
        logging.debug("Started swap round")
        driver_run_requests = {}
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            driver_run_requests[run_id] = set()
            for stop in manifest[current_order:]:
                booking_id = stop["booking_id"]
                if stop["action"] == "pickup":
                    driver_run_requests[run_id].add(booking_id)
        
        SwapHandler.manifest_options = []
        initial_cost = 0
        pool = mp.Pool(self.MAX_NUM_THREAD)
        for driver_run in self.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            run_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
            active_requests_in_manifest = driver_run_requests[run_id]
            manifest_cost = SwapHandler.get_manifest_cost(driver_run)
            initial_cost += manifest_cost
            SwapHandler.manifest_options.append((run_id,active_requests_in_manifest,manifest_cost,driver_run))

            for booking_id in active_requests_in_manifest:
                driver_run_without_request = SwapHandler.remove_request_from_driver_run(driver_run, booking_id, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                cost = SwapHandler.get_manifest_cost(driver_run_without_request)
                requests_in_new_manifest = active_requests_in_manifest.copy()
                requests_in_new_manifest.remove(booking_id)
                SwapHandler.manifest_options.append((run_id, requests_in_new_manifest, cost, driver_run_without_request))

                for other_booking_id in self.active_requests:
                    if other_booking_id in active_requests_in_manifest:
                        continue
                    request = self.request_dic[other_booking_id]
                    requests_after = requests_in_new_manifest.copy()
                    requests_after.add(other_booking_id)

                    args = (self.depot, run_id, requests_after, driver_run_without_request, request, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                    pool.apply_async(SwapHandler.create_manifest_option, args=args, callback=SwapHandler.process_result)

                    # cost, driver_run_with_new_request = SwapHandler.insert_request_to_driver_run(
                    #     self.depot, driver_run_without_request, request, self.DWELL_PICKUP, self.DWELL_ALIGHT)
                    # SwapHandler.manifest_options.append((run_id, requests_after, cost, driver_run_with_new_request))

        pool.close()
        pool.join()

        # Filter out invalid manifest options
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
        logging.debug("Number of manifest options: {0}".format(no_options))
        logging.debug("Started building optimization problem")
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
                logging.info("Total time spent on optimization: {0}".format(m.Runtime))

                for i in range(no_options):
                    if x_t[i].X == 1:
                        selected_options.append(SwapHandler.manifest_options[i])

            else:
                raise Exception("Gurobi solver ended with code: {0}".format(m.Status))
        
        new_cost = 0
        no_of_swaps = 0
        selected_driver_runs = {}
        for run_id, active_requests, cost, driver_run in selected_options:
            new_cost += cost
            selected_driver_runs[run_id] = driver_run
            prev_requests_in_manifest = driver_run_requests[run_id]
            uncommon_items = prev_requests_in_manifest.symmetric_difference(active_requests)
            num_uncommon = len(uncommon_items)
            no_of_swaps += num_uncommon
        no_of_swaps //= 2  # Each swap is counted twice (once for each driver run)
        logging.info('Initial cost: {0}, new cost: {1}, cost reduction'.format(initial_cost, new_cost, initial_cost-new_cost))
        logging.info('Number of swaps: {0}'.format(no_of_swaps))

        self.new_driver_runs = []
        run_ids = list(selected_driver_runs.keys())
        run_ids.sort()
        for run_id in run_ids:
            driver_run = selected_driver_runs[run_id]
            self.new_driver_runs.append(driver_run)

        return self.new_driver_runs, initial_cost-new_cost, no_of_swaps

    def create_manifest_option(depot_node, run_id, requests, driver_run, request, DWELL_PICKUP, DWELL_ALIGHT):
        cost, new_driver_run = SwapHandler.insert_request_to_driver_run(
            depot_node, driver_run, request, DWELL_PICKUP, DWELL_ALIGHT)
        return (run_id, requests, cost, new_driver_run)

    def process_result(result):
        SwapHandler.manifest_options.append(result)

    def get_manifest_cost(driver_run):
        cost = 0

        state = driver_run[PayloadParser.DRIVER_STATE]
        no_completed_stops = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        remaining_stops = driver_run[PayloadParser.DRIVER_MANIFEST][no_completed_stops:]

        current_node = SwapHandler.stop_to_node(state)
        for stop in remaining_stops:
            next_node = SwapHandler.stop_to_node(stop)
            cost += NetworkHandler.travel_time(current_node,next_node)
            current_node = next_node
        return cost

    def stop_to_node(stop):
        return Node(stop["loc"]["lat"],stop["loc"]["lon"],id=stop["loc"]["node_id"])

    def remove_request_from_driver_run(driver_run, booking_id, DWELL_PICKUP, DWELL_ALIGHT):
        state = copy.deepcopy(driver_run[PayloadParser.DRIVER_STATE])
        no_completed_stops = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        menifest = copy.deepcopy(driver_run[PayloadParser.DRIVER_MANIFEST])
        new_manifest = menifest[:no_completed_stops]
        remaining_stops = []

        current_node = SwapHandler.stop_to_node(state)
        current_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
        current_order = no_completed_stops
        for stop in menifest[no_completed_stops:]:
            if stop["booking_id"] == booking_id:
                continue
            stop_node = SwapHandler.stop_to_node(stop)
            travel_time = NetworkHandler.travel_time(current_node, stop_node)
            current_time += travel_time
            if current_time < stop["time_window_start"]:
                current_time = stop["time_window_start"]
            stop["scheduled_time"] = current_time
            current_order += 1
            stop["order"] = current_order

            if stop["action"] == "pickup":
                current_time += DWELL_PICKUP
            else:
                current_time += DWELL_ALIGHT

            remaining_stops.append(stop)
            current_node = stop_node

        state[PayloadParser.DRIVER_STATE_T_LOCS] = len(new_manifest)
        
        return {
            PayloadParser.DRIVER_STATE: state,
            PayloadParser.DRIVER_MANIFEST: new_manifest + remaining_stops
        }
        

    def insert_request_to_driver_run(depot_node, driver_run, request, DWELL_PICKUP, DWELL_ALIGHT):
        driver_run_c = copy.deepcopy(driver_run)

        pickup_stop = {'run_id': None, 'booking_id': request['booking_id'], 'order': -1, 'action': "pickup", 
            "loc": request["pickup_pt"], 'scheduled_time': -1, 
            'am': request["am"], 'wc': request["wc"], 'time_window_start': request['pickup_time_window_start'],
            'time_window_end': request['pickup_time_window_end']}
        dropoff_stop = {'run_id': None, 'booking_id': request['booking_id'], 'order': -1, 'action': "dropoff",
            "loc": request["dropoff_pt"], 'scheduled_time': -1, 
            'am': request["am"], 'wc': request["wc"], 'time_window_start': request['dropoff_time_window_start'],
            'time_window_end': request['dropoff_time_window_end']}

        load = 0
        state = driver_run_c[PayloadParser.DRIVER_STATE]
        pickup_stop["run_id"] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        dropoff_stop["run_id"] = state[PayloadParser.DRIVER_STATE_RUN_ID]
        manifest = driver_run_c[PayloadParser.DRIVER_MANIFEST]
        start_node = SwapHandler.stop_to_node(state)
        start_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
        completed_stops = []
        remaining_stops = []
        for stop in manifest:
            if stop["order"] <= state[PayloadParser.DRIVER_STATE_LOC_SERV]:
                if stop["action"] == "pickup":
                    load += stop["am"]
                else:
                    load -= stop["am"]
                completed_stops.append(stop)
            else:
                remaining_stops.append(stop)
        

        end_time = state["end_time"]
        objective = "vmt"
        args_list = [(i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective,depot_node, end_time, DWELL_PICKUP, DWELL_ALIGHT) 
                     for i in range(len(remaining_stops) + 1) 
                     for j in range(i + 1, len(remaining_stops) + 2)]
        
        results = [SwapHandler.evaluate_insertion(args) for args in args_list]

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
        return best_cost,new_driver_run


    def evaluate_insertion(args):
        i, j, remaining_stops, pickup_stop, dropoff_stop, start_time, start_node, load, state, objective, depot, end_time, dwell_pickup, dwell_alight = args
        new_manifest = copy.deepcopy(remaining_stops[:i] + [pickup_stop] + remaining_stops[i:j] + [dropoff_stop] + remaining_stops[j:])
        current_time = start_time
        current_node = start_node
        current_load = load
        cost = 0
        order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        index = 0
        for stop in new_manifest:
            next_node = SwapHandler.stop_to_node(stop)
            travel_time = NetworkHandler.travel_time(current_node, next_node)
            cost += travel_time
            current_node = next_node
            current_time += travel_time
            if current_time < stop["time_window_start"]:
                current_time = stop["time_window_start"]
            stop["scheduled_time"] = current_time
            if objective == "pick_up_time" and (i == index or j == index):
                stop["time_window_end"] = current_time + 30
            if current_time > stop["time_window_end"]:
                return float("inf"), None
            if stop["action"] == "pickup":
                current_load += stop["am"]
                current_time += dwell_pickup
            else:
                current_load -= stop["am"]
                current_time += dwell_alight
            if current_load > state["am_capacity"]:
                return float("inf"), None
            order += 1
            stop["order"] = order
            index += 1

        if current_time+NetworkHandler.travel_time(current_node,depot) > end_time:
            return float("inf"), None
        if objective == "pick_up_time":
            return new_manifest[i]["scheduled_time"], new_manifest
        return cost, new_manifest
