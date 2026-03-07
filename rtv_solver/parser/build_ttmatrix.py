import numpy as np
import json
import os
import pickle

from rtv_solver.handlers.network_handler import NetworkHandler

input_directory = 'inputs/test_nc'
output_dir = "inputs/test_nc/ttm"

files_to_process = ["test_12r_1v_repeat9.json"]

for filename in files_to_process:
    input_path = os.path.join(input_directory, filename)

    # Skip if the file doesn't exist
    if not os.path.isfile(input_path):
        print(f"Skipping {filename}, not found in input directory.")
        continue
    
    print("Processing file: ", filename)
    with open(input_path, 'rb') as file:
        payload_wilson_initial = json.load(file)

    NetworkHandler.init(True, "http://127.0.0.1:5001/", euclidean=False)
    depot = payload_wilson_initial["depot"]
    depot_node_id = NetworkHandler.get_next_node_id(depot["pt"]["lat"],depot["pt"]["lon"])
    depot["pt"]["node_id"] = depot_node_id
    
    for request in payload_wilson_initial["requests"]:
        node_id = NetworkHandler.get_next_node_id(request["pickup_pt"]["lat"], request["pickup_pt"]["lon"])
        request["pickup_pt"]["node_id"] = node_id
        
        node_id = NetworkHandler.get_next_node_id(request["dropoff_pt"]["lat"], request["dropoff_pt"]["lon"])
        request["dropoff_pt"]["node_id"] = node_id
    
    for driver_run in payload_wilson_initial["driver_runs"]:
        node_id = NetworkHandler.get_next_node_id(driver_run["state"]["loc"]["lat"], driver_run["state"]["loc"]["lon"])
        driver_run["state"]["loc"]["node_id"] = node_id
        for stop in driver_run["manifest"]:
            if "loc" in stop:
                node_id = NetworkHandler.get_next_node_id(stop["loc"]["lat"], stop["loc"]["lon"])
                stop["loc"]["node_id"] = node_id
    
    travel_time_matrix, no_of_nodes, SERVER_BASED, EUCLIDEAN = NetworkHandler.initialize_travel_time_matrix()
    # Convert RawArray to np.array
    travel_time_matrix_np = np.frombuffer(travel_time_matrix, dtype=np.float64).reshape((int(no_of_nodes.value), int(no_of_nodes.value)))
    
    payload_wilson_initial["travel_time_matrix"] = travel_time_matrix_np.tolist()
    
    os.makedirs(output_dir, exist_ok=True)
    # Define the full paths for the new files
    json_path = os.path.join(output_dir, filename)
    pkl_path = os.path.join(output_dir, filename.replace('.json', '.pkl'))
    
    # Write JSON file
    with open(json_path, 'w') as f:
        json.dump(payload_wilson_initial, f)
    # Write pickle file
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload_wilson_initial, f)