import pickle
import argparse

from rtv_solver import OnlineRTVSolver, OfflineRTVSolver

DEBUG_MODE = True
# TODO current code only works with wilson-format due to the keys that are being used in the dictionaries to handle data

if __name__ == "__main__":
    """Code is a normal Python script based on the jupyter-notebook in order to be able to debug it more easily."""
    # TODO move config management to YAML config or Hydra
    parser = argparse.ArgumentParser(description='Arguments for the RTV solver main script')
    # technical setup
    parser.add_argument('--output_dir', type=str,           default="output_format/debug/", help='output directory')
    parser.add_argument('--input_file', type=str,           default="rtv-solver/inputs/wilson_nc_initial.pkl", help='Request file')
    parser.add_argument('--server_url', type=str,           default="http://127.0.0.1:5001/", help='Server URL')
    parser.add_argument('--max_thread_cnt', type=int,       default=64, help='Maximum thread count for parallel processing')
    parser.add_argument('--rtv_timeout', type=int,          default=120, help='RTV construction timeout in seconds')
    parser.add_argument('--ilp_timeout', type=int,          default=120, help='ILP solver timeout in seconds')
    parser.add_argument('--ilp_penalty', type=int,              default=1000000, help='Penalty for not serving a trip')
    # experiment parameters
    parser.add_argument('--max_cardinality', type=int,      default=2, help='Maximum trips to be shared when creating trips') # alt: total trips in same vehicle
    parser.add_argument('--largest_tsp', type=int,          default=8, help='Largest TSP to be solved when constructing RTVs') # incl existing passengers
    parser.add_argument('--share_cost_factor', type=int,    default=10, help='Shareable cost factor in [???]')
    parser.add_argument('--rebalancing', type=bool,         default=False, help='Whether to enable rebalancing of vehicles')
    parser.add_argument('--dwell_pickup', type=int,         default=180, help='Dwell time at pickup in seconds')
    parser.add_argument('--dwell_alight', type=int,         default=60, help='Dwell time at alight (dropoff) in seconds')
    parser.add_argument('--rh_factor', type=int,            default=0, help='Rolling horizon factor') # NOTE alternative step_size
    parser.add_argument('--step_size', type=int,            default=600, help='Step size in seconds for rolling horizon')
    parser.add_argument('--batch_interval', type=int,       default=3600, help='Batch interval in seconds') # NOTE still used?
    # TODO COAML parameters 
    config = parser.parse_args()

    # load data from 
    file = open(config.input_file, 'rb')
    data = pickle.load(file)
    file.close()

    if DEBUG_MODE: # check if the basic functionality of the online RTV solver works (foundation for offline RTV solver)
        # reduce the complexity by only considering a single vehicle
        driver_runs_total = data["driver_runs"]
        driver_runs_reduced = driver_runs_total[:1]

        # create a simplified set of requests, consider all requests that start before end_requests
        current_time = 5*3600 + 30*60
        step = 10*60
        selected_requests = []
        for request in data["requests"]:
            if request["pickup_time_window_start"] < current_time + step:
                selected_requests.append(request)

        # create a new payload with selected requests
        new_payload = {
            "depot": data["depot"],
            "requests": selected_requests,
            "driver_runs": driver_runs_reduced}
        # Initialize RTV solver 
        on_solver = OnlineRTVSolver(
            config)
        new_driver_runs, unserved_requests = on_solver.solve_pdptw_rtv(new_payload)
        print(f"No. unserved requests: {unserved_requests}")
    else:
        off_solver = OfflineRTVSolver(config)
        driver_runs, unserved_requests = off_solver.solve_rtv(data, config.batch_interval, config.step_size)
