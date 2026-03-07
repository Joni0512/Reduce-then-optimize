# import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import hexaly.optimizer
from .handlers.network_handler import NetworkHandler
from .structure.node import Node

import gurobipy as gp
from gurobipy import GRB

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

# TODO implement this code as a replacement solver for the gurobi solver
# https://github.com/DMadhuranga/rtv-solver/commit/583212865d4d96c8b882e56ecf3033ae74026251 to see relevant changes
# TODO clean up code and make sure it works
# TODO remove parts of code that are not needed for the solver or that are already part of other classes, seems to be a lot of repetition from payload etc.

class GurobiSolver:
    """
    Gurobi solver for solving the PDPTW in one go and not making use of the RTV procedure.

    Installation through (https://www.gurobi.com/solutions/licensing/). Academic License is required.

    Currently does not work as some of the conditions are incorrect. 
    """
    # TODO add init that handles the config values
    def __init__(self, config):
        self.config = config

    @staticmethod
    def solve_pdptw(server_url, payload, time_limit, output_dir, iteration, min_truck=False, dwell_pickup=180, dwell_dropoff=60, tt_matrix=None):
        console_logger.info(f"Solving PDPTW with Hexaly solver")
        
        time_matrix, no_of_nodes = None, None
        NetworkHandler.init_from_source(server_url=server_url, tt_matrix=tt_matrix)

        depot = payload["depot"]
        if tt_matrix == None:
            depot_node_id = NetworkHandler.get_next_node_id(depot["pt"]["lat"],depot["pt"]["lon"])
        else:
            depot_node_id = depot["pt"]["node_id"]
        depot_node = Node(depot["pt"]["lat"],depot["pt"]["lon"], id=depot_node_id)
        
        requests: list = copy.deepcopy(payload["requests"])
        unserved = []
        # start with all requests being unserverd
        # NOTE problem if we use this to solve from situation in-between, as the solver will not know which requests are already served or are boarded etc.
        for request in requests:
            unserved.append(request["booking_id"])
        nb_customers = len(requests)*2 # each request has a pickup and a dropoff
        nb_requests = len(requests)

        driver_runs = copy.deepcopy(payload["driver_runs"])
        truck_capacity = driver_runs[0]["state"]["am_capacity"] # NOTE sets capacity equal for all trucks
        
        # initialise data structures for the solver
        unused_trucks = []
        used_trucks = []

        fixed_dropoff_requests = {}
        active_requests = set()
        request_pickup_map = {}
        request_dropoff_map = {}
        fixed_requests = [] # does not influence solver but is used to print the fixed requests in the output file
        
        completed_pickups = set()
        completed_dropoffs = set()
        truck_load_data = []
        truck_current_time_data = []
        truck_current_location_data = []

        for i in range(len(driver_runs)):
            if driver_runs[i]["state"]["locations_already_serviced"] == 0:
                unused_trucks.append(i)
            else:
                used_trucks.append(i)
        for driver_run in driver_runs:
            state = driver_run["state"]
            manifest = driver_run["manifest"]

            # Updating node_id for current location
            state_loc = state["loc"]
            if tt_matrix is None:
                node_id = NetworkHandler.get_next_node_id(state_loc["lat"],state_loc["lon"])
                state_loc["node_id"] = node_id
            state["loc"] = state_loc

            completed = driver_run["state"]["locations_already_serviced"]

            # Updating node_id for remaining stops
            for stop in manifest[completed:]:
                if tt_matrix is None:
                    node_id = NetworkHandler.get_next_node_id(stop["loc"]["lat"],stop["loc"]["lon"])
                    stop["loc"]["node_id"] = node_id
                active_requests.add(stop["booking_id"])
                if stop["action"] == "pickup":
                    request_pickup_map[stop["booking_id"]] = stop
                else: # dropoff
                    request_dropoff_map[stop["booking_id"]] = stop
                    if stop["booking_id"] not in request_pickup_map:
                        for stop_t in manifest[:completed]:
                            if stop_t["booking_id"] == stop["booking_id"]:
                                request_pickup_map[stop["booking_id"]] = stop_t
                                break

            load = 0
            nb_customers += len(manifest) # add the overall solution based on existing manifest stops + new requests (see above)
            for stop in manifest[:completed]:
                nb_customers -= 1
                # figure out current load of the truck
                if stop["action"] == "pickup":
                    load += stop["am"]
                    fixed_dropoff_requests[stop["booking_id"]] = stop["run_id"]
                    fixed_requests.append(stop["booking_id"])
                    completed_pickups.add(stop["booking_id"])
                else:
                    load -= stop["am"]
                    del fixed_dropoff_requests[stop["booking_id"]]
                    completed_dropoffs.add(stop["booking_id"])
            
            truck_load_data.append(load)
            truck_current_time_data.append(driver_run["state"]["location_dt_seconds"])
            truck_current_location_data.append(driver_run["state"]["loc"]["node_id"])
        
        nb_trucks = len(driver_runs)
        max_horizon = driver_runs[0]["state"]["end_time"]

        console_logger.info(f"Truck load data: {truck_load_data}")
        console_logger.info(f"No of customers: {nb_customers}")
        console_logger.info(f"No of requests: {nb_requests}")
        console_logger.info(f"No of trucks: {nb_trucks}")
        console_logger.info(f"Max horizon: {max_horizon}")

        if tt_matrix is None:
            # TODO move this further up in this call to order responsibilities better
            # add nodes to NetworkHandler travel_time_matrix for easier calculation
            for request in requests:
                origin = request["pickup_pt"]
                dest = request["dropoff_pt"]
                origin_id = NetworkHandler.get_next_node_id(origin["lat"],origin["lon"])
                origin["node_id"] = origin_id
                dest_id = NetworkHandler.get_next_node_id(dest["lat"],dest["lon"])
                dest["node_id"] = dest_id

        for booking_id in active_requests:
            # TODO move this up once the code is cleaned up and responsibilities are ordered better
            # add active requests (in-progress, not yet completed) from manifest to the main requests list
            request = {}
            pickup_stop = request_pickup_map[booking_id]
            dropoff_stop = request_dropoff_map[booking_id]
            request = {
                "pickup_pt":                pickup_stop["loc"],
                "pickup_time_window_start": pickup_stop["time_window_start"],
                "pickup_time_window_end":   pickup_stop["time_window_end"],
                "dropoff_pt":               dropoff_stop["loc"],
                "am":                       dropoff_stop["am"],
                "dwell_pickup":             dwell_pickup,
                "dwell_alight":             dwell_dropoff,
                "dropoff_time_window_start":dropoff_stop["time_window_start"],
                "dropoff_time_window_end":  dropoff_stop["time_window_end"],
                "booking_id":               booking_id,
            }
            requests.append(request)

        nb_requests += len(active_requests)

        # --- Solver index mapping and per-node data arrays ---
        # node_map / reverse_node_map translate between network node IDs and the 0-based integer indices the Hexaly solver expects.
        # reverse_request_map allows tracing a solver node back to its originating request.
        node_map = {}
        reverse_node_map = {}
        reverse_request_map = {}
        node_start = 0 # monotonically increasing index for the solver nodes
        # Pre-allocated arrays indexed by solver node index (size = nb_customers)
        demands_data = [0] * nb_customers # positive = pickup load, negative = dropoff load
        service_time_data = [0] * nb_customers # dwell time at stop
        earliest_start_data = [0] * nb_customers # time window lower bound
        latest_end_data = [0] * nb_customers # time window upper bound
        # Sentinel: -1 = this node IS a pickup, >=0 = index of paired pickup node, -2 = pickup already completed
        pick_up_index = [0] * nb_customers
        # Sentinel: -1 = this node IS a dropoff, >=0 = index of paired dropoff node
        delivery_index = [0] * nb_customers
        # -1 = unassigned (free for solver), run_id = dropoff locked to this specific vehicle run
        run_id_fix_data = [0] * nb_customers
        request_index = 0
        
        for request in requests:
            booking_id = request["booking_id"]
            origin_id = request["pickup_pt"]["node_id"]
            dest_id = request["dropoff_pt"]["node_id"]

            # Register solver nodes for each request, skipping fully completed ones. If a pickup is already done (passenger on-board), only the dropoff node is added and its run_id is fixed to prevent the solver from re-assigning it.
            if booking_id not in completed_dropoffs: 
                # skip completed requests
                origin = node_start
                if booking_id not in completed_pickups: 
                    # skip pickup when dropoff is still part of route
                    node_map[origin_id] = origin
                    reverse_node_map[origin] = origin_id
                    reverse_request_map[origin] = request_index
                    node_start += 1
                    demands_data[origin] = request["am"]
                    service_time_data[origin] = dwell_pickup
                    earliest_start_data[origin] = request["pickup_time_window_start"]
                    latest_end_data[origin] = request["pickup_time_window_end"]
                    pick_up_index[origin] = -1
                    run_id_fix_data[origin] = -1
                    delivery_index[origin] = node_start

                dest = node_start
                node_map[dest_id] = dest
                reverse_node_map[dest] = dest_id
                reverse_request_map[dest] = request_index
                node_start += 1
                demands_data[dest] = -request["am"]
                service_time_data[dest] = dwell_dropoff
                earliest_start_data[dest] = request["dropoff_time_window_start"]
                latest_end_data[dest] = request["dropoff_time_window_end"]
                
                # Register dropoff node; link back to pickup or mark as pre-boarded
                if booking_id not in completed_pickups:
                    pick_up_index[dest] = origin # paired with its pickup
                    run_id_fix_data[dest] = -1  # no fixed vehicle assignment
                else:
                    pick_up_index[dest] = -2 # pickup already completed
                    run_id_fix_data[dest] = fixed_dropoff_requests[request["booking_id"]]  # lock to assigned run
                delivery_index[dest] = -1
            request_index += 1

        if tt_matrix is None:
            time_matrix, no_of_nodes, _, _ = NetworkHandler.initialize_travel_time_matrix()
        no_of_nodes = int(no_of_nodes.value)
        time_matrix = np.frombuffer(time_matrix, dtype=np.float64)
        time_matrix = time_matrix.reshape((no_of_nodes, no_of_nodes))
        

        # distances between depot and among each customer node ((nb_customers+1) × (nb_customers+1))
        dist_matrix_data = [[0] * (nb_customers+1) for i in range(nb_customers+1)]
        depot_id = depot_node.id
        for i in range(nb_customers):
            for j in range(nb_customers):
                dist_matrix_data[i+1][j+1] = time_matrix[reverse_node_map[i]][reverse_node_map[j]]

        for i in range(nb_customers):
            dist_matrix_data[0][i+1] = time_matrix[depot_id][reverse_node_map[i]]
        for i in range(nb_customers):
            dist_matrix_data[i+1][0] = time_matrix[reverse_node_map[i]][depot_id]

        # distance matrix from each node to each truck position (nb_trucks × (nb_customers+1))
        dist_matrix_from_start_data = [[0] * (nb_customers + 1) for i in range(nb_trucks)]
        for i in range(nb_trucks):
            dist_matrix_from_start_data[i][0] = time_matrix[truck_current_location_data[i]][depot_id]
            for j in range(nb_customers):
                dist_matrix_from_start_data[i][j+1] = time_matrix[truck_current_location_data[i]][reverse_node_map[j]]

        pickup_node = []
        delivery_node = []

        for i in range(nb_customers):
            if pick_up_index[i] == -1:  # i is pickup
                pickup_node.append(i)
                delivery_node.append(delivery_index[i])

        nb_requests = len(pickup_node)
        R = range(nb_requests)

        console_logger.info(f"Distance matrix shape: {len(dist_matrix_data)},{len(dist_matrix_data[0])}")
        console_logger.info(f"Distance matrix: {dist_matrix_data}")
        console_logger.info(f"No of demand data: {len(demands_data)}")
        routes = {}

        if output_dir:
            # ensure the file exists
            os.makedirs(output_dir / "hexaly", exist_ok=True)
            with open(output_dir / "hexaly" / "fixed_requests_{0}.txt".format(iteration), 'a') as f:
                f.write("Iteration	{0} :  {1}\n".format(iteration," ".join(str(x) for x in fixed_requests)))

        # TODO debug test
        model = gp.Model("VRPPD_TW")

        # -------------------------
        # Sets
        # -------------------------
        K = range(nb_trucks)
        N = range(nb_customers)
        R = range(nb_requests)

        # Include depot as node 0
        V = range(nb_customers + 1)  # 0 = depot
        customers = range(1, nb_customers + 1)

        # -------------------------
        # Decision Variables
        # -------------------------

        # x[k,i,j] = 1 if truck k goes from i to j
        x = model.addVars(K, V, V, vtype=GRB.BINARY, name="x")

        # customer served by truck k
        y = model.addVars(K, customers, vtype=GRB.BINARY, name="y")

        # arrival time
        t = model.addVars(K, customers, vtype=GRB.CONTINUOUS, lb=0, name="t")

        # request served
        # z[r] = 1 → request r is fully served; z[r] = 0 → request skipped
        z = model.addVars(R, vtype=GRB.BINARY, name="request_served")

        # cumulative load
        load = model.addVars(K, customers, vtype=GRB.CONTINUOUS, lb=0, name="load")

        # truck used
        truck_used = model.addVars(K, vtype=GRB.BINARY, name="truck_used")

        # lateness per customer
        # lateness = model.addVars(K, customers, vtype=GRB.CONTINUOUS, lb=0, name="lateness")

        # home lateness
        # home_lateness = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0, name="home_lateness")

        # -------------------------
        # Constraints
        # -------------------------

        # No self loops
        for k in K:
            for i in V:
                model.addConstr(x[k, i, i] == 0)

        # # Each customer visited exactly once
        # for j in customers:
        #     model.addConstr(gp.quicksum(y[k, j] for k in K) == 1)

        # Each customer visited at most once
        for j in customers:
            model.addConstr(gp.quicksum(y[k, j] for k in K) <= 1)

        # Link y and x
        for k in K:
            for j in customers:
                model.addConstr(
                    gp.quicksum(x[k, i, j] for i in V) == y[k, j]
                )
                model.addConstr(
                    gp.quicksum(x[k, j, i] for i in V) == y[k, j]
                )

        # Truck used
        for k in K:
            model.addConstr(
                truck_used[k] >= gp.quicksum(y[k, j] for j in customers) / nb_customers
            )

        # Flow from depot
        for k in K:
            model.addConstr(
                gp.quicksum(x[k, 0, j] for j in customers) == truck_used[k]
            )
            model.addConstr(
                gp.quicksum(x[k, j, 0] for j in customers) == truck_used[k]
            )

        # ----------
        # Request pairing
        # ------------------
        for r in R:
            p = pickup_node[r] + 1
            d = delivery_node[r] + 1

            # pickup served iff request served
            model.addConstr(y[k, p] - y[k, d] == 0)

        model.addConstr(
            gp.quicksum(y[k, p] for k in K) == z[r]
        )

        model.addConstr(
            gp.quicksum(y[k, d] for k in K) == z[r]
        )

        # sanity check
        for r in R:
            assert pick_up_index[delivery_node[r]] == pickup_node[r]        

        # -------------------------
        # Capacity accumulation
        # -------------------------

        M = 1e6

        for k in K:
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            load[k, j] >= load[k, i]
                            + demands_data[j-1]
                            - M * (1 - x[k, i, j])
                        )

            for j in customers:
                model.addConstr(load[k, j] <= truck_capacity)
                model.addConstr(load[k, j] >= demands_data[j-1] * y[k, j])

        # -------------------------
        # Time propagation
        # -------------------------

        for k in K:
            # Depot → first customer propagation
            for j in customers:
                model.addConstr(
                    t[k, j] >=
                    truck_current_time_data[k]
                    + dist_matrix_from_start_data[k][j]
                    - M * (1 - x[k, 0, j])
                )

            # Customer → customer propagation
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            t[k, j] >=
                            t[k, i]
                            + service_time_data[i-1]
                            + dist_matrix_data[i][j]
                            - M * (1 - x[k, i, j])
                        )
        # # -------------------------
        # # Time windows
        # # -------------------------

        # for k in K:
        #     for j in customers:
        #         model.addConstr(t[k, j] >= earliest_start_data[j-1])
        #         model.addConstr(
        #             lateness[k, j] >= t[k, j] - latest_end_data[j-1]
        #         )

        # -------------------------
        # Hard Time Windows
        # -------------------------
        for k in K:
            for j in customers:

                # Enforce only if customer is served by truck k
                model.addConstr(
                    t[k, j] >= earliest_start_data[j-1] - M * (1 - y[k, j])
                )

                model.addConstr(
                    t[k, j] <= latest_end_data[j-1] + M * (1 - y[k, j])
                )    
        # -------------------------
        # Pickup & Delivery
        # -------------------------

        for i in N:
            if pick_up_index[i] == -1:
                d = delivery_index[i]

                for k in K:
                    model.addConstr(
                        y[k, i+1] == y[k, d+1]
                    )

                    model.addConstr(
                        t[k, d+1] >= t[k, i+1] + service_time_data[i]
                    )

            if run_id_fix_data[i] != -1:
                fixed_k = run_id_fix_data[i]
                model.addConstr(y[fixed_k, i+1] == 1)

        # # -------------------------
        # # Home lateness
        # # -------------------------

        # for k in K:
        #     for i in customers:
        #         model.addConstr(
        #             home_lateness[k] >=
        #             t[k, i]
        #             + service_time_data[i-1]
        #             + dist_matrix_data[i][0]
        #             - max_horizon
        #             - M * (1 - x[k, i, 0])
        #         )

        # -------------------------
        # Hard return before max_horizon
        # -------------------------

        for k in K:
            for i in customers:
                model.addConstr(
                    t[k, i]
                    + service_time_data[i-1]
                    + dist_matrix_data[i][0]
                    <= max_horizon
                    + M * (1 - x[k, i, 0])
                )

        # -------------------------
        # Objective
        # -------------------------

        # total_lateness = (
        #     gp.quicksum(lateness[k, j] for k in K for j in customers)
        #     + gp.quicksum(home_lateness[k] for k in K)
        # )

        # model.setObjective(total_lateness, GRB.MINIMIZE)

        total_distance = gp.quicksum(
            dist_matrix_data[i][j] * x[k, i, j]
            for k in K
            for i in V
            for j in V
        )

        # primary: Maximize number of served requests
        model.ModelSense = GRB.MAXIMIZE
        # First objective: maximize served requests
        model.setObjectiveN(
            gp.quicksum(z[r] for r in R),
            index=0,
            priority=2
        )

        # Second objective: minimize distance
        # Since global sense is MAXIMIZE, minimize distance by maximizing negative distance
        model.setObjectiveN(
            - total_distance,
            index=1,
            priority=1
        )

        model.setObjective(total_distance, GRB.MINIMIZE)

        model.Params.TimeLimit = time_limit

        # FIXME debug
        model.addConstr(z[0] == 1)

        model.optimize()

        # Write the solution in a file with the following format:
        #  - number of trucks used and total distance
        #  - for each truck the customers visited (omitting the start/end at the depot)
            
        
        routes = {}

        for k in K:
            routes[k] = []

            if truck_used[k].X < 0.5:
                continue

            # find first customer after depot
            current = None
            for j in V:
                if j != 0 and x[k, 0, j].X > 0.5:
                    current = j
                    break

            # follow path
            while current is not None and current != 0:
                customer_index = current - 1  # convert back to 0..nb_customers-1
                routes[k].append(customer_index)

                next_node = None
                for j in V:
                    if x[k, current, j].X > 0.5:
                        next_node = j
                        break

                if next_node == 0:
                    break

                current = next_node
        
        # creating driver runs from routes
        new_driver_runs = []
        for driver_run in driver_runs:
            state = copy.deepcopy(driver_run["state"])
            # state["locations_already_serviced"] = 0
            # state["location_dt_seconds"] = 0
            # state["loc"] = depot
            new_driver_run = {"state": state, "manifest": driver_run["manifest"][:state["locations_already_serviced"]]}
            new_driver_runs.append(new_driver_run)
        for route_no, route in routes.items():
            driver_run = new_driver_runs[route_no]
            driver_run["state"]["total_locations"] = len(route) + driver_run["state"]["locations_already_serviced"]
            current_time = driver_run["state"]["location_dt_seconds"]
            current_loc = driver_run["state"]["loc"]["node_id"]
            current_order = driver_run["state"]["locations_already_serviced"] + 1
            for i in route:
                console_logger.info(f"Current order: {current_order}, Current location: {current_loc}, Current time: {current_time}")
                current_time += time_matrix[current_loc][reverse_node_map[i]]
                if current_time < earliest_start_data[i]:
                    current_time = earliest_start_data[i]
                action = "pickup" if pick_up_index[i] == -1 else "dropoff"
                request = requests[reverse_request_map[i]]
                booking_id = request["booking_id"]
                demand = demands_data[i] if pick_up_index[i] == -1 else - demands_data[i]
                loc = request["pickup_pt"] if action == "pickup" else request["dropoff_pt"]
                stop = {'run_id': driver_run["state"]["run_id"], 'booking_id': booking_id, 'order': current_order, 'action': action, 
                        "loc": loc, 'scheduled_time': current_time, 
                        'am': demand, 'wc': 0, 'time_window_start': earliest_start_data[i], 
                        'time_window_end':latest_end_data[i], 'dwell': service_time_data[i]}
                current_loc = reverse_node_map[i]    
                current_order += 1
                current_time += service_time_data[i]
                driver_run["manifest"].append(stop)
            # driver_run["state"]["end_time"] = current_time      

        if output_dir:
            # ensure the file exists
            os.makedirs(output_dir / "hexaly", exist_ok=True)
            with open(output_dir / "hexaly" / "sol_{0}.txt".format(iteration), 'w') as f:
                for driver_run in new_driver_runs:
                    manifest = driver_run["manifest"]
                    nodes = []
                    for stop in manifest:
                        nodes.append(str(int(stop["loc"]["node_id"])))
                    nodes = " ".join(nodes)
                    if len(manifest) > 0:
                        f.write("Route	{0} :  {1}\n".format(driver_run["state"]["run_id"],nodes))
                    # route_id += 1
        return new_driver_runs, []