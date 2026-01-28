import pickle
import argparse
import logging

from rtv_solver import OnlineRTVSolver, OfflineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser

DEBUG_MODE = True # reduces number of vehicles and requests for easier debugging
ONLINE_MODE = False # runs all requests in one go without rolling horizon batching
# TODO current code only works with wilson-format due to the keys that are being used in the dictionaries to handle data
# TODO add logging again
if __name__ == "__main__":
    """Main script to run RTV solver in online or offline mode based on input data and configuration parameters with easy options for debugging and testing."""
    # TODO move config management to YAML config or Hydra
    parser = argparse.ArgumentParser(description='Arguments for the RTV solver main script')
    # technical setup
    parser.add_argument('--output_dir', type=str,           default="output_format/debug/", help='output directory')
    parser.add_argument('--input_file', type=str,           default="rtv-solver/inputs/wilson_nc_initial.pkl", help='Request file') 
    # alternative: rtv-solver/inputs/localDB_payload_oct.pkl
    parser.add_argument('--server_url', type=str,           default="http://127.0.0.1:5001/", help='Server URL')
    parser.add_argument('--max_thread_cnt', type=int,       default=16, help='Maximum thread count for parallel processing')
    parser.add_argument('--rtv_timeout', type=int,          default=120, help='RTV construction timeout in seconds')
    parser.add_argument('--ilp_timeout', type=int,          default=120, help='ILP solver timeout in seconds')
    parser.add_argument('--ilp_penalty', type=int,          default=1000000, help='Penalty for not serving a trip')
    # experiment parameters
    parser.add_argument('--max_cardinality', type=int,      default=2, help='Maximum trips to be shared when creating trips') # alt: total trips in same vehicle
    parser.add_argument('--largest_tsp', type=int,          default=8, help='Largest TSP to be solved when constructing RTVs') # incl existing passengers
    parser.add_argument('--share_cost_factor', type=int,    default=10, help='Shareable cost factor in [???]')
    parser.add_argument('--rebalancing', type=bool,         default=False, help='Whether to enable rebalancing of vehicles')
    parser.add_argument('--dwell_pickup', type=int,         default=180, help='Dwell time at pickup in seconds')
    parser.add_argument('--dwell_alight', type=int,         default=60, help='Dwell time at alight (dropoff) in seconds')
    parser.add_argument('--rh_factor', type=int,            default=0, help='Rolling horizon factor') # NOTE alternative to step_size, still used?
    parser.add_argument('--step_size', type=int,            default=300, help='Step size in seconds for rolling horizon')
    parser.add_argument('--batch_interval', type=int,       default=3600, help='Batch interval in seconds')
    # TODO COAML parameters 
    config = parser.parse_args()

    # load data from file and update to canonical format for the entire system
    file = open(config.input_file, 'rb')
    data = pickle.load(file)
    file.close()
    data = PayloadParser.normalize_to_canonical(data)

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.output_dir + 'main.log'),
        logging.StreamHandler()]
        )   
    logging.info(f' --- Start: RTV simulation with step size // batch interval: {config.step_size} // {config.batch_interval}')

    if DEBUG_MODE: # check if the basic functionality of the online RTV solver works (foundation for offline RTV solver)
        logging.getLogger().setLevel(logging.DEBUG)
        config.rtv_timeout = 600000 # if I am clicking through inputs, it never breaks due to timeout
        
        # reduce the complexity by only considering a single vehicle
        driver_runs_total = data[PayloadParser.DRIVERS]
        driver_runs_reduced = driver_runs_total[:1]
        # create a simplified set of requests, consider all requests that start before end_requests
        current_time = 5*3600 + 30*60
        step = 20*60
        selected_requests = []
        for request in data[PayloadParser.REQUESTS]:
            if request[PayloadParser.REQ_PICKUP_WINDOW_START] < current_time + step:
                selected_requests.append(request)
        # create a new payload with selected requests
        payload = {
            PayloadParser.DEPOT: data[PayloadParser.DEPOT],
            PayloadParser.REQUESTS: selected_requests,
            PayloadParser.DRIVERS: driver_runs_reduced}
    else: 
        payload = data

    # Initialize RTV solver
    if ONLINE_MODE:
        on_solver = OnlineRTVSolver(config)
        updated_driver_runs, unserved_requests = on_solver.solve_pdptw_rtv(payload)
    else:
        off_solver = OfflineRTVSolver(config)
        updated_driver_runs, unserved_requests = off_solver.solve_rtv(payload, config.batch_interval, config.step_size)

    logging.info(f"\033[1m {len(unserved_requests)}\033[0m unserved requests with IDs:\n{unserved_requests}")
