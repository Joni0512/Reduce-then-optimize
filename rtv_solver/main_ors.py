import pickle
import numpy as np
import os
import pprint
from rtv_solver import OnlineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser

# INPUT
file_path = "rtv-solver/inputs/wilson_nc_initial.pkl"
# Current code only works with wilson format due to the keys that are being used

if __name__ == "__main__":
    """took the code from the notebook and build a normal script that can be debugged."""
    file = open(file_path, 'rb')
    wilson_bool = False
    if file_path.split("/")[1].split("_")[0] == "wilson":
        wilson_bool = True
    data = pickle.load(file)
    file.close()

    # NOTE for debugging only consider a single vehicle
    if True:
        driver_runs_total = data[PayloadParser.DRIVERS]
        driver_runs_reduced = driver_runs_total[:1]
        data[PayloadParser.DRIVERS][:] = driver_runs_reduced

    # TODO add arg
    # Initialize the RTV solver with the URL of the OSRM server
    online_rtv_solver = OnlineRTVSolver("http://127.0.0.1:5001/")

    # creating a new payload with new requests
    # consider all requests that start before 05:40:00
    current_time = 5*3600+30*60
    step = 10*60
    selected_requests = []
    for request in data["requests"]:
        if request["pickup_time_window_start"] < current_time+step:
            selected_requests.append(request)

    # create a new payload with the selected requests
    new_payload = {
        "depot": data["depot"],
        "requests": selected_requests,
        "driver_runs": data["driver_runs"]}

    ## Fast Heuristic method
    new_driver_runs, unserved_requests = online_rtv_solver.solve_pdptw_rtv(new_payload)
    print(f"No. unserved requests: {unserved_requests}")