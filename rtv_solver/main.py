import argparse
import logging
import time
import json

from pathlib import Path

from rtv_solver import OnlineRTVSolver, OfflineRTVSolver, COAMLPipeline

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser

from rtv_solver.structure.config import Config

from rtv_solver.util.logger import setup_loggers, BASIC_LOGGER, DATA_LOGGER
from rtv_solver.util.helper import save_json, set_seed

from rtv_solver.visuals.payload_visuals import plot_requests_operating_area
from rtv_solver.visuals.route_manifest_mapper import RouteManifestMapper

if __name__ == "__main__":
    """
    Main script to run RTV solver in online or offline mode based on input data and configuration parameters with easy options for debugging and testing.
    
    Assumptions (wilson - 02.02.2026):
    - All vehicles start from the same location (depot) and need to return there at the end of their shift.
    - Accepted waiting times are already defined in the request payload and currently not defined by the program (30 min between earliest and latest pickup - travel_time defines allowed dropoff times) 
    - The first accepted request of a vehicle is directly considered as boarded (rationale: one vehicle has to commit trip and thus it is already fixed with that vehicle, second request is only considered boarded.)
    - Rebalancing: Rejected requests are in underserved areas, so we need to send additional vehicles there.
    """
    # TODO move config management to YAML config or Hydra
    parser = argparse.ArgumentParser(description='Arguments for the RTV solver main script')
    # reproduction setup
    parser.add_argument('--config_file', type=str,          default="", help='Path to JSON config file instead of defining the other values manually')
    parser.add_argument('--override', action='append',      default=[], help='Override config values when loading from a config file, e.g. key=value (can be repeated)')
    # technical parameters
    parser.add_argument('--output_dir', type=str,           default="debug", help='Output directory')
    parser.add_argument('--input_file', type=str,           default="wilson_nc_initial.pkl", help='Request input file') # alternative: rtv-solver/inputs/localDB_payload_oct.pkl
    parser.add_argument('--server_url', type=str,           default="http://127.0.0.1:5001/", help='Backend server URL')
    parser.add_argument('--max_thread_cnt', type=int,       default=16, help='Maximum thread count for parallel processing')
    parser.add_argument('--rtv_timeout', type=int,          default=120, help='RTV construction timeout in seconds')
    parser.add_argument('--ilp_timeout', type=int,          default=120, help='ILP solver timeout in seconds')
    parser.add_argument('--ilp_penalty', type=int,          default=100_000, help='Penalty for not serving a trip')
    # experiment parameters
    parser.add_argument('--max_cardinality', type=int,      default=2, help='Maximum trips to be shared when creating trips in one batch_interval') # alt: total trips in same vehicle
    parser.add_argument('--largest_tsp', type=int,          default=8, help='Largest TSP to be solved when constructing RTVs') # incl existing passengers
    parser.add_argument('--share_cost_factor', type=int,    default=1.2, help='Shareable cost factor in factor of original single cost [???]') # TODO originally the value was 10, that value is extremely high and thus too many trips are considered feasible (ideally we would apply this earlier to reduce the amount of trips / tripCosts generated)
    parser.add_argument('--rebalancing', type=bool,         default=False, help='(NOT WOKRING 12.02.2026) Vehicles are rebalanced if the need arises based on missed requests and idling vehicles.')
    parser.add_argument('--keep_active', type=bool,         default=True, help='Active requests from an ILP solution in a prior iteration must be kept.')
    parser.add_argument('--return_depot', type=bool,        default=True, help="Vehicles must return to the originating depot.")
    parser.add_argument('--dwell_pickup', type=int,         default=180, help='Dwell time at pickup in seconds')
    parser.add_argument('--dwell_alight', type=int,         default=60, help='Dwell time at alight (dropoff) in seconds')
    parser.add_argument('--walk_distance_cutoff', type=int, default=0, help="Walking distance between dropoff and final destination.")
    parser.add_argument('--step_size', type=int,            default=1800, help='Step size in seconds for rolling horizon')
    parser.add_argument('--batch_interval', type=int,       default=3600, help='Batch interval in seconds')
    # stats parameters
    parser.add_argument('--travel_time_margin', type=int,   default=5, help='Error margin for travel time in stats calculation')
    # TODO COAML parameters 
    # random_seed, training parameters, NN parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') 
    parser.add_argument('--mode', '-m', type=str, choices=['online', 'offline', 'rh-ml', 'plot'], default='rh-ml', help='Mode on how the programme should solve the PDPTW')
    parser.add_argument('--debug', type=bool, default=True, help='Run in debug mode (# reduces number of vehicles and requests for easier debugging)')

    # implement configuration
    arguments = parser.parse_args()
    config = Config.from_args(arguments)

    setup_loggers(config.OUTPUT_DIR)
    console_logger = logging.getLogger(BASIC_LOGGER)
    data_logger = logging.getLogger(DATA_LOGGER)

    console_logger.info(f"Output directory: {config.OUTPUT_DIR}")
    console_logger.info(f' --- Start: RTV simulation --- online > {config.MODE}')
    console_logger.info(f'Arguments: {config}')

    # load data from file and update to canonical format for the entire system
    data = PayloadParser.load_input_data(Path(__file__).resolve().parent.parent / "inputs" / config.INPUT_FILE)

    set_seed(config.SEED, config.DEBUG)
    
    if config.DEBUG: # check if the basic functionality of the online RTV solver works (foundation for offline RTV solver)
        console_logger.setLevel(logging.INFO)
        config.RTV_TIMEOUT = 600000 # if I am clicking through inputs, it never breaks due to timeout
        
        # reduce the complexity by only considering a single vehicle
        driver_runs_total = data[PayloadParser.DRIVERS]
        driver_runs_reduced = driver_runs_total[:1] 
        # test to change the first vehicle to trigger certain situations
        vehicle_state = driver_runs_reduced[0][PayloadParser.DRIVER_STATE]
        vehicle_manifest = driver_runs_reduced[0][PayloadParser.DRIVER_MANIFEST]        
        # vehicle_state[PayloadParser.DRIVER_STATE_END_TIME] = 25000
        config.MAX_CARDINALITY = 3

        # BUG combination 2 --> iteration keeps running and still tries to optimize despite no active vehicle being left
        # TODO how to set vehicles to inactive, so they are not part of the optimization anymore but are also completed in their manifest (depot return and complete manifest of prior assigned trips)
        # vehicle_state[PayloadParser.DRIVER_STATE_END_TIME] = 22000 
        config.RETURN_DEPOT = True
        config.KEEP_ACTIVE = True

        # combination 3 
        # if trip is not considered in recent trips but is the last dropoff (situation: new trip is injected before that last dropoff in a new iteration)
        # BUG find situation where this issue rises and build a test from it, relevant for multiple issues
        
        # create a simplified set of requests, consider all requests that start before end_requests
        current_time = 5*3600 + 30*60
        step = 5*60
        selected_requests = []
        for request in data[PayloadParser.REQUESTS]:
            if request[PayloadParser.REQ_PICKUP_WINDOW_START] < current_time + step:
                selected_requests.append(request)

        # combination 3 not yet implemented
        # TODO create a payload where rebalancing is needed (minimal wait time between pickup window start and end, multiple requests and multiple vehicles)
        
        # create a new payload with selected requests
        payload = {
            PayloadParser.DEPOT: data[PayloadParser.DEPOT],
            PayloadParser.REQUESTS: selected_requests,
            PayloadParser.DRIVERS: driver_runs_reduced}
    else: 
        payload = data

    if config.MODE != 'plot':
        # Initialize RTV solver
        start_time = time.time()
        if config.MODE == 'online':
            on_solver = OnlineRTVSolver(config)
            updated_driver_runs, _ = on_solver.solve_pdptw_rtv(payload)
        elif config.MODE == 'offline':
            off_solver = OfflineRTVSolver(config)
            updated_driver_runs = off_solver.solve_rtv(payload, config.BATCH_INTERVAL, config.STEP_SIZE)
        elif config.MODE == 'rh-ml':
            rh_solver = COAMLPipeline(config)
            updated_driver_runs = rh_solver.solve_pdptw(payload)
        else:
            updated_driver_runs = []
            console_logger.info('No solution')
            
        # calculate statistics of each iteration; for now only the first vehicle
        stats_payload = {PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
                        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
                        PayloadParser.DRIVERS: updated_driver_runs}
        stats_evaluator = StatsParser(config)
        feasible, stats, violations = stats_evaluator.evaluate(stats_payload)
        assignment_history = stats_evaluator.evaluate_development(stats_payload)
        
        console_logger.info(stats)
        console_logger.info(f'Violations: {violations}')
        console_logger.info(f"Total time: {time.time() - start_time}")

        console_logger.info("Request history analysed.")
        console_logger.info(assignment_history)

        save_json(stats_payload, 
                config.OUTPUT_DIR / "result_driver_runs.json")
        save_json({"stats": stats, "violations": violations},
                config.OUTPUT_DIR / "results.json")

        console_logger.info(f"Run complete. Results can be found @ {Path(config.OUTPUT_DIR)}")
    
    # VISUALISE
        with open(config.OUTPUT_DIR / "result_driver_runs.json", 'r') as driver_runs_file:
            loaded_data = json.load(driver_runs_file)
        mapper = RouteManifestMapper(config)
        geojson = mapper.manifest_to_geojson(loaded_data, 18)
        mapper.save_geojson(geojson, config.OUTPUT_DIR / "route_manifest.geojson")

    plot_requests_operating_area(payload, show=False, save_path=config.OUTPUT_DIR / "request_distribution.png") 
    
