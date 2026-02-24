# import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse
import copy
import hexaly.optimizer
import sys
import math
import random
import pickle
from .handlers.network_handler import NetworkHandler
from .handlers.payload_parser import PayloadParser
from .structure.node import Node
from hexaly.optimizer import HxCallbackType

# TODO implement this code as a replacement solver for the gurobi solver
# https://github.com/DMadhuranga/rtv-solver/commit/583212865d4d96c8b882e56ecf3033ae74026251 to see relevant changes
# TODO remove parts of code that are not needed for the solver or that are already part of other classes, seems to be a lot of repetition from payload etc.
# TODO install hexaly to run this solver (https://www.hexaly.com/docs/last/installation/installationonmacosx.html)
# TODO P10 (no priority)

class HexalySolver:

    def solve_pdptw(server_url, payload, time_limit, output_dir, iteration, min_truck=False, dwell_pickup=180, dwell_dropoff=60, tt_matrix=None):

        time_matrix, no_of_nodes = None, None
        if tt_matrix == None:           
            NetworkHandler.init(True, server_url)
        else:
            time_matrix, no_of_nodes, _, _ = NetworkHandler.init(False, server_url, tt_matrix)

        depot = payload["depot"]
        if tt_matrix == None:
            depot_node_id = NetworkHandler.get_next_node_id(depot["pt"]["lat"],depot["pt"]["lon"])
        else:
            depot_node_id = depot["pt"]["node_id"]
        depot_node = Node(depot["pt"]["lat"],depot["pt"]["lon"], id=depot_node_id)

        
        requests = copy.deepcopy(payload["requests"])
        unserved = []
        for request in requests:
            unserved.append(request["booking_id"])
        driver_runs = copy.deepcopy(payload["driver_runs"])
        nb_customers = len(requests)*2
        truck_capacity = driver_runs[0]["state"]["am_capacity"]
        completed_pickups = set()
        completed_dropoffs = set()
        truck_load_data = []
        truck_current_time_data = []
        truck_current_location_data = []
        fixed_dropoff_requests = {}
        unused_trucks = []
        used_trucks = []

        active_requests = set()
        request_pickup_map = {}
        request_dropoff_map = {}

        fixed_requests = []
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
                else:
                    request_dropoff_map[stop["booking_id"]] = stop
                    if stop["booking_id"] not in request_pickup_map:
                        for stop_t in manifest[:completed]:
                            if stop_t["booking_id"] == stop["booking_id"]:
                                request_pickup_map[stop["booking_id"]] = stop_t
                                break

            load = 0
            nb_customers += len(manifest)
            for stop in manifest[:completed]:
                nb_customers -= 1
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

        # print(truck_load_data)

        node_map = {}
        reverse_node_map = {}
        reverse_request_map = {}
        node_start = 0

        demands_data = [0] * nb_customers
        service_time_data = [0] * nb_customers
        earliest_start_data = [0] * nb_customers
        latest_end_data = [0] * nb_customers
        pick_up_index = [0] * nb_customers
        delivery_index = [0] * nb_customers
        run_id_fix_data = [0] * nb_customers
        request_index = 0

        if tt_matrix is None:
            for request in requests:
                origin = request["pickup_pt"]
                dest = request["dropoff_pt"]
                origin_id = NetworkHandler.get_next_node_id(origin["lat"],origin["lon"])
                origin["node_id"] = origin_id
                dest_id = NetworkHandler.get_next_node_id(dest["lat"],dest["lon"])
                dest["node_id"] = dest_id

        for booking_id in active_requests:
            request = {}
            pickup_stop = request_pickup_map[booking_id]
            request["pickup_pt"] = pickup_stop["loc"]
            request["pickup_time_window_start"] = pickup_stop["time_window_start"]
            request["pickup_time_window_end"] = pickup_stop["time_window_end"]
            dropoff_stop = request_dropoff_map[booking_id]
            request["dropoff_pt"] = dropoff_stop["loc"]
            request["am"] = dropoff_stop["am"]
            request["dwell_pickup"] = dwell_pickup
            request["dwell_alight"] = dwell_dropoff
            request["dropoff_time_window_start"] = dropoff_stop["time_window_start"]
            request["dropoff_time_window_end"] = dropoff_stop["time_window_end"]
            request["booking_id"] = booking_id
            requests.append(request)

        for request in requests:
            booking_id = request["booking_id"]
            origin_id = request["pickup_pt"]["node_id"]
            dest_id = request["dropoff_pt"]["node_id"]

            if booking_id not in completed_dropoffs:
                origin = node_start
                if booking_id not in completed_pickups:
                    node_map[origin_id] = origin
                    reverse_node_map[origin] = origin_id
                    reverse_request_map[origin] = request_index
                    node_start += 1
                    demands_data[origin] = request["am"]
                    service_time_data[origin] = dwell_pickup
                    earliest_start_data[origin] = request["pickup_time_window_start"]
                    latest_end_data[origin] = request["pickup_time_window_end"]
                    pick_up_index[origin] = -1
                    delivery_index[origin] = node_start
                    run_id_fix_data[origin] = -1
                dest = node_start
                node_map[dest_id] = dest
                reverse_node_map[dest] = dest_id
                reverse_request_map[dest] = request_index
                node_start += 1
                demands_data[dest] = -request["am"]
                service_time_data[dest] = dwell_dropoff
                earliest_start_data[dest] = request["dropoff_time_window_start"]
                latest_end_data[dest] = request["dropoff_time_window_end"]
                if booking_id not in completed_pickups:
                    pick_up_index[dest] = origin
                    run_id_fix_data[dest] = -1
                else:
                    pick_up_index[dest] = -2
                    run_id_fix_data[dest] = fixed_dropoff_requests[request["booking_id"]]
                delivery_index[dest] = -1
            request_index += 1

        if tt_matrix is None:
            time_matrix, no_of_nodes, _, _ = NetworkHandler.initialize_travel_time_matrix()
        no_of_nodes = int(no_of_nodes.value)
        time_matrix = np.frombuffer(time_matrix, dtype=np.float64)
        time_matrix = time_matrix.reshape((no_of_nodes, no_of_nodes))
        
        print("No of customers: ",nb_customers)
        dist_matrix_data = [[0] * (nb_customers+1) for i in range(nb_customers+1)]
        depot_id = payload["depot"]["node_id"]
        for i in range(nb_customers):
            for j in range(nb_customers):
                dist_matrix_data[i+1][j+1] = time_matrix[reverse_node_map[i]][reverse_node_map[j]]

        for i in range(nb_customers):
            dist_matrix_data[0][i+1] = time_matrix[depot_id][reverse_node_map[i]]
        for i in range(nb_customers):
            dist_matrix_data[i+1][0] = time_matrix[reverse_node_map[i]][depot_id]
        # nb_customers, nb_trucks, truck_capacity, dist_matrix_data, \
        #     demands_data, service_time_data, earliest_start_data, latest_end_data, \
        #     pick_up_index, delivery_index, max_horizon = read_input_pdptw(instance_file)

        dist_matrix_from_start_data = [[0] * (nb_customers + 1) for i in range(nb_trucks)]
        for i in range(nb_trucks):
            dist_matrix_from_start_data[i][0] = time_matrix[truck_current_location_data[i]][depot_id]
            for j in range(nb_customers):
                dist_matrix_from_start_data[i][j+1] = time_matrix[truck_current_location_data[i]][reverse_node_map[j]]

        print("Distance matrix shape: ", len(dist_matrix_data),len(dist_matrix_data[0]))
        print("No of demand data: ",len(demands_data))
        routes = {}

        if output_dir:
            with open(output_dir+"fixed_requests.txt".format(iteration), 'a') as f:
                f.write("Iteration	{0} :  {1}\n".format(iteration," ".join(str(x) for x in fixed_requests)))

        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            #
            # Declare the optimization model
            #
            model = optimizer.model

            # Sequence of customers visited by each truck
            customers_sequences = [model.list(nb_customers) for k in range(nb_trucks)]

            # All customers must be visited by exactly one truck
            model.constraint(model.partition(customers_sequences))

            # /Create Hexaly arrays to be able to access them with "at" operators
            demands = model.array(demands_data)
            earliest = model.array(earliest_start_data)
            latest = model.array(latest_end_data)
            service_time = model.array(service_time_data)
            dist_matrix = model.array(dist_matrix_data)
            dist_matrix_from_start = model.array(dist_matrix_from_start_data)
            truck_loads = model.array(truck_load_data)
            truck_current_time = model.array(truck_current_time_data)
            # run_id_fix = model.array(run_id_fix_data)
            # dist_depot = model.array(dist_depot_data)

            dist_routes = [None] * nb_trucks
            end_time = [None] * nb_trucks
            home_lateness = [None] * nb_trucks
            lateness = [None] * nb_trucks

            # A truck is used if it visits at least one customer
            new_trucks_used = [] 
            for k in range(nb_trucks):
                if k not in used_trucks:
                    new_trucks_used.append(model.count(customers_sequences[k]) > 0)
            trucks_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_trucks)]
            nb_trucks_used = model.sum(new_trucks_used)

            # Pickups and deliveries
            customers_sequences_array = model.array(customers_sequences)
            for i in range(nb_customers):
                if pick_up_index[i] == -1:
                    pick_up_list_index = model.find(customers_sequences_array, i)
                    delivery_list_index = model.find(customers_sequences_array, delivery_index[i])
                    model.constraint(pick_up_list_index == delivery_list_index)
                    pick_up_list = model.at(customers_sequences_array, pick_up_list_index)
                    delivery_list = model.at(customers_sequences_array, delivery_list_index)
                    model.constraint(model.index(pick_up_list, i) < model.index(delivery_list, delivery_index[i]))
                if run_id_fix_data[i] != -1:
                    drop_off_list_index = model.find(customers_sequences_array, i)
                    model.constraint(drop_off_list_index == run_id_fix_data[i])

            for k in range(nb_trucks):
                sequence = customers_sequences[k]
                c = model.count(sequence)

                # The quantity needed in each route must not exceed the truck capacity at any
                # point in the sequence
                demand_lambda = model.lambda_function(
                    lambda i, prev: prev + demands[sequence[i]])
                route_quantity = model.array(model.range(0, c), demand_lambda, truck_loads[k])

                quantity_lambda = model.lambda_function(
                    lambda i: route_quantity[i] <= truck_capacity)
                model.constraint(model.and_(model.range(0, c), quantity_lambda))

                # Distance traveled by each truck
                dist_lambda = model.lambda_function(
                    lambda i: model.at(dist_matrix, sequence[i - 1]+1, sequence[i]+1))
                depot_dist_lambda = model.lambda_function(
                    lambda i: model.at(dist_matrix, 0, i+1))
                dist_routes[k] = model.sum(model.range(1, c), dist_lambda) \
                    + model.iif(c > 0, model.at(dist_matrix_from_start, k, sequence[0] + 1) + model.at(dist_matrix,sequence[c - 1]+1,0),model.at(dist_matrix_from_start, k, 0))

                # End of each visit
                end_lambda = model.lambda_function(
                    lambda i, prev:
                        model.max(
                            earliest[sequence[i]],
                            model.iif(
                                i == 0,
                                truck_current_time[k] + model.at(dist_matrix_from_start, k, sequence[0]+1),
                                prev + model.at(dist_matrix, sequence[i - 1]+1, sequence[i]+1)))
                        + service_time[sequence[i]])

                end_time[k] = model.array(model.range(0, c), end_lambda, 0)

                # Arriving home after max_horizon
                home_lateness[k] = model.iif(
                    trucks_used[k],
                    model.max(
                        0,
                        end_time[k][c - 1] + model.at(dist_matrix,sequence[c - 1]+1,0) - max_horizon),
                    0)

                # Completing visit after latest_end
                late_selector = model.lambda_function(
                    lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
                lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_selector)

            # Total lateness (must be 0 for the solution to be valid)
            total_lateness = model.sum(lateness)

            # Total distance traveled
            total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)

            # Objective: minimize the number of trucks used, then minimize the distance traveled
            model.minimize(total_lateness)
            if min_truck:
                model.minimize(nb_trucks_used)
                model.minimize(total_distance)
            else:
                model.minimize(total_distance)
                model.minimize(nb_trucks_used)
                
            model.close()

            # adding existing schedule
            for i in range(nb_trucks):
                driver_run = driver_runs[i]
                manifest = driver_run["manifest"]
                completed = driver_run["state"]["locations_already_serviced"]
                customer_seq_val = customers_sequences[i].get_value()
                customer_seq_val.clear()
                for stop in manifest[completed:]:
                    node_id = stop["loc"]["node_id"]
                    if node_id in node_map:
                        node = node_map[node_id]
                        customer_seq_val.add(node)

            # Parameterize the optimizer
            optimizer.param.time_limit = time_limit

            # optimizer.save_environment("outputs_hexaly/export.hxb.gz")
            # optimizer.write("outputs_hexaly/model.lp")
            optimizer.solve()

            #
            # Write the solution in a file with the following format:
            #  - number of trucks used and total distance
            #  - for each truck the customers visited (omitting the start/end at the depot)
            #
            # print(customers_sequences)
            # print("%d %.2f\n" % (nb_trucks_used.value, total_distance.value))
            # with open(sol_file, 'w') as f:
            
            # if total_lateness.value > 0:
            #     return payload["driver_runs"], unserved
            route_no = 0
            print_no = 0
            for k in range(nb_trucks):
                routes[route_no] = []
                # Values in sequence are in 0...nbCustomers. +2 is to put it back in
                # 2...nbCustomers+2 as in the data files (1 being the depot)
                if k in used_trucks or trucks_used[k].value == 1:
                    # f.write("Route %d: " % print_no)
                    for customer in customers_sequences[k].value:
                        routes[route_no].append(customer)
                        # f.write("%d " % (customer + 1))
                    print_no += 1
                # f.write("\n")
                route_no += 1
            
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
                    # print("Current order: ", current_order)
                    # print("Current location: ", current_loc)
                    # print("Current time: ", current_time)
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
            with open(output_dir+"sol_{0}.txt".format(iteration), 'w') as f:
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