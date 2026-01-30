import pickle
import argparse
import logging
import time
import json

from rtv_solver import OnlineRTVSolver, OfflineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.structure.config import Config

DEBUG_MODE = True # reduces number of vehicles and requests for easier debugging
ONLINE_MODE = False # runs all requests in one go without rolling horizon batching

def setup_logging():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.output_dir + 'main.log'),
        logging.StreamHandler()]
        ) 
    # TODO this does not work, it still spams my debug message
    logging.getLogger("requests").setLevel(logging.WARNING)

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
    parser.add_argument('--walk_distance_cutoff', type=int, default=0, help="Walking distance between dropoff and final destination.")
    # parser.add_argument('--rh_factor', type=int,            default=0, help='Rolling horizon factor')  # NOTE alternative to step_size
    parser.add_argument('--step_size', type=int,            default=300, help='Step size in seconds for rolling horizon')
    parser.add_argument('--batch_interval', type=int,       default=1200, help='Batch interval in seconds')
    # stats parameters
    parser.add_argument('--travel_time_margin', type=int,   default=5, help='Error margin for travel time in stats calculation')
    # TODO COAML parameters 
    arguments = parser.parse_args()
    config = Config.from_args(arguments)

    # load data from file and update to canonical format for the entire system
    file = open(config.input_file, 'rb')
    data = pickle.load(file)
    file.close()
    data = PayloadParser.normalize_to_canonical(data)
    
    setup_logging()
    logging.info(f' --- Start: RTV simulation online {ONLINE_MODE}')
    logging.info(f'Arguments: {config}')

    if DEBUG_MODE: # check if the basic functionality of the online RTV solver works (foundation for offline RTV solver)
        logging.getLogger().setLevel(logging.INFO)
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
    start_time = time.time()
    if ONLINE_MODE:
        on_solver = OnlineRTVSolver(config)
        updated_driver_runs, unserved_requests = on_solver.solve_pdptw_rtv(payload)
    else:
        off_solver = OfflineRTVSolver(config)
        updated_driver_runs, unserved_requests = off_solver.solve_rtv(payload, config.batch_interval, config.step_size)

    # calculate statistics of each iteration; for now only the first vehicle
    stats_payload = {PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
                        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
                        PayloadParser.DRIVERS: updated_driver_runs}
    stats_evaluator = StatsParser()
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload, unserved_requests)
    
    logging.info(f"Stats: {json.dumps(stats.to_dict(), indent=4)}")
    logging.info(f'Violations: {violations}')
    logging.info(f"{len(unserved_requests)} unserved requests")
    logging.info(f"Total time: {time.time() - start_time}")
