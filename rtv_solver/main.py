import argparse
import logging
import time
import json

from pathlib import Path

from rtv_solver import OnlineRTVSolver, OfflineRTVSolver, COAMLPipeline, HexalySolver
from rtv_solver.training_loop import COAMLTrainingLoop

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser

from rtv_solver.schema.payload_keys import PayloadKeys

from rtv_solver.structure.config import Config

from rtv_solver.util.logger import setup_loggers, BASIC_LOGGER, DATA_LOGGER
from rtv_solver.util.helper import save_json, set_seed

from rtv_solver.visuals.route_manifest_mapper import RouteManifestMapper

from rtv_solver.pipeline import TYPE_BEST_ORDERED_MATCH, TYPE_BEST_UNORDERED_MATCH

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
    parser.add_argument('--server_url', type=str,           default="http://127.0.0.1:5001/", help='Backend server URL')
    parser.add_argument('--max_thread_cnt', type=int,       default=16, help='Maximum thread count for parallel processing')
    parser.add_argument('--rtv_timeout', type=int,          default=120, help='RTV construction timeout in seconds')
    parser.add_argument('--ilp_timeout', type=int,          default=120, help='ILP solver timeout in seconds')
    parser.add_argument('--ilp_penalty', type=int,          default=100_000, help='Penalty for not serving a trip')
    # experiment parameters
    parser.add_argument('--max_cardinality', type=int,      default=5, help='Maximum trips to be shared when creating trips in one batch_interval') # alt: total trips in same vehicle
    parser.add_argument('--largest_tsp', type=int,          default=16, help='Largest TSP to be solved when constructing RTVs') # incl existing passengers
    parser.add_argument('--share_cost_factor', type=int,    default=5, help='Shareable cost factor in factor of original single cost [???]') # TODO originally the value was 10, that value is extremely high and thus too many trips are considered feasible (ideally we would apply this earlier to reduce the amount of trips / tripCosts generated)
    parser.add_argument('--rebalancing', type=str,          default='False', choices=['True', 'False'], help='(NOT WOKRING 12.02.2026) Vehicles are rebalanced if the need arises based on missed requests and idling vehicles.')
    parser.add_argument('--keep_active', type=str,          default='False', choices=['True', 'False'], help='Active requests from an ILP solution in a prior iteration must be kept.')
    parser.add_argument('--return_depot', type=str,         default='True', choices=['True', 'False'], help="Vehicles must return to the originating depot.")
    parser.add_argument('--intermediate_location', type=str, default='False', choices=['True', 'False'], help='Intermediate locations are considered for the solver.')
    parser.add_argument('--dwell_pickup', type=int,         default=0, help='Dwell time at pickup in seconds')
    parser.add_argument('--dwell_alight', type=int,         default=0, help='Dwell time at alight (dropoff) in seconds')
    # TODO add dwell time again as we have taken it out for specific testing purposes
    parser.add_argument('--walk_distance_cutoff', type=int, default=0, help="Walking distance between dropoff and final destination.")
    parser.add_argument('--step_size', type=int,            default=200, help='Step size in seconds for rolling horizon')
    parser.add_argument('--batch_interval', type=int,       default=400, help='Batch interval in seconds') # NOTE if this value is too small, we might miss requests if the vehicle_trip to the pickup is longer than the batch interval size (TODO fix this so this does not have as much impact)
    # TODO time is very different across liLim and wilson, so we need to adjust the values accordingly
    # stats parameters
    parser.add_argument('--travel_time_margin', type=int,   default=5, help='Error margin for travel time in stats calculation')
    # random_seed, training parameters, NN parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') 
    parser.add_argument('--mode', '-m', type=str, choices=['online', 'offline', 'coaml', 'plot', 'optimal_solution', 'hexaly_solution'], default='coaml', help='Mode on how the programme should solve the PDPTW')
    parser.add_argument('--debug', type= str, default='True', choices=['True', 'False'], help='Run in debug mode (# reduces number of vehicles and requests for easier debugging)')
    parser.add_argument('--y_star_type', type=str, choices=[TYPE_BEST_ORDERED_MATCH, TYPE_BEST_UNORDERED_MATCH], default=TYPE_BEST_ORDERED_MATCH, help='Type of y_star to be used for the Fenchel-Young loss during imitation learning')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs over the same payload for COAML mode')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='Learning rate for the ML model')

    # TODO FIXME fix input file handling for Sartori datasets (result does not work as expected and has many violations) - no priority to fix this at end of thesis
    # parser.add_argument('--input_file', type=str,           default="sartori/n100/bar-n100-1.txt", help='Request input file') # does not require backend server as time matrix is available in dataset
    # TODO FIXME fix chattanooga input (violations after offline run) - no priority to fix this at end of thesis
    # parser.add_argument('--input_file', type=str,           default="localDB_payload_oct.pkl", help='Request input file')  # chattanooga dataset
    
    parser.add_argument('--input_file', type=str,           default="solutions/li_lim/manifests/lc101.json", help='Request input file') # alternative: rtv-solver/inputs/localDB_payload_oct.pkl;inputs/
    # parser.add_argument('--input_file', type=str,           default="test_nc/ttm/test_10r_1v_repeat6_simple.pkl", help='Request input file') 
    # parser.add_argument('--input_file', type=str,           default="wilson_nc_initial.pkl", help='Request input file') 

    # TODO make sure when COAML is selected, the right optimal solution is also selected to run the experiments automatically with the right setttings
    parser.add_argument('--imitation_solution_file', type=str, default='solutions/li_lim/manifests/lc101.json', help='Path to the imitation solution file with the complete manifest of all trips for all vehicles')
    # implement configurations
    arguments = parser.parse_args()
    config = Config.from_args(arguments)
    config.enforce_constraints()

    setup_loggers(config.OUTPUT_DIR)
    console_logger = logging.getLogger(BASIC_LOGGER)
    data_logger = logging.getLogger(DATA_LOGGER)

    console_logger.info(f"Input file: {config.INPUT_FILE}")
    console_logger.info(f"Output directory: {config.OUTPUT_DIR}")
    console_logger.info(f' --- Start: RTV simulation --- online > {config.MODE}')
    console_logger.info(f'Arguments: {config}')

    # load data from file and update to canonical format for the entire system
    data = PayloadParser.load_input_data(Path(__file__).resolve().parent.parent / config.INPUT_FILE)

    set_seed(config.SEED, config.DEBUG)
    
    if config.DEBUG: # check if the basic functionality of the online RTV solver works (foundation for offline RTV solver)
        console_logger.setLevel(logging.INFO)
        config.RTV_TIMEOUT = 600000 # if I am clicking through inputs, it never breaks due to timeout
        
        # reduce the complexity by only considering a single vehicle
        driver_runs_total = data[PayloadKeys.DRIVERS]
        driver_runs_reduced = driver_runs_total[:1] 
        # test to change the first vehicle to trigger certain situations
        vehicle_state = driver_runs_reduced[0][PayloadKeys.DRIVER_STATE]
        vehicle_manifest = driver_runs_reduced[0][PayloadKeys.DRIVER_MANIFEST]        
        # vehicle_state[PayloadKeys.DRIVER_STATE_END_TIME] = 25000

        # FIXME --> iteration keeps running and still tries to optimize despite no active vehicle being left
        # TODO how to set vehicles to inactive, so they are not part of the optimization anymore but are also completed in their manifest (depot return and complete manifest of prior assigned trips)

        # NOTE not able to reproduce bug below but might be fixed with other issues
        # if trip is not considered in recent trips but is the last dropoff (situation: new trip is injected before that last dropoff in a new iteration)
        # BUG find situation where this issue rises and build a test from it, relevant for multiple issues
        
        # create a simplified set of requests, consider all requests that start before end_requests
        # TIME-BASED SELECTION OF REQUESTS
        # current_time = 5*3600 + 30*60
        # step = 5*60
        # selected_requests = []
        # for request in data[PayloadKeys.REQUESTS]:
        #     if request[PayloadKeys.REQ_PICKUP_WINDOW_START] < current_time + step:
        #         selected_requests.append(request)

        # COUNT_BASED SELECTION OF REQUESTS: add code that just takes the first n requests out of the payload
        selected_requests = data[PayloadKeys.REQUESTS][:10]

        # create a new payload with selected requests
        payload = {
            PayloadKeys.DEPOT: data[PayloadKeys.DEPOT],
            PayloadKeys.REQUESTS: selected_requests,
            PayloadKeys.DRIVERS: driver_runs_reduced}
    else: 
        payload = data

    # required to run data from other sources without the backend server
    # TODO refactoring: this code is currently badly designed as for each change of the payload, we need to handle time_matrix separately (high priority refactoring)
    if data.get(PayloadKeys.TIME_MATRIX) is not None:
        payload[PayloadKeys.TIME_MATRIX] = data[PayloadKeys.TIME_MATRIX]
    else:
        payload[PayloadKeys.TIME_MATRIX] = None
        console_logger.warning("Time matrix is not available. Solution run on server, but time_matrix is missing - leading to no possibility of running this dataset without backend server.") 

    if config.MODE != 'plot':
        # Initialize RTV solver
        start_time = time.time()
        if config.MODE == 'online':
            on_solver = OnlineRTVSolver(config)
            updated_driver_runs, _ = on_solver.solve_pdptw_rtv(payload)
        elif config.MODE == 'offline':
            off_solver = OfflineRTVSolver(config)
            updated_driver_runs = off_solver.solve_rtv(payload, config.BATCH_INTERVAL, config.STEP_SIZE)
        elif config.MODE == 'coaml':
            if config.EPOCHS > 1:
                training_loop = COAMLTrainingLoop(config, payload)
                training_result = training_loop.run()
                updated_driver_runs = training_result.updated_driver_runs
                print(f"Epoch iteration losses: {training_result.epoch_iteration_losses}")
                print(f"All iteration losses: {training_result.all_iteration_losses}")
            else:
                # NOTE experiment with a single loop and later run the loop with multiple files based on a folder that is iterated over
                cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

                rh_solver = COAMLPipeline(config, cleared_payload)
                updated_driver_runs = rh_solver.solve_pdptw(cleared_payload)
                print(f"Loss history: {rh_solver.loss_history}")
        elif config.MODE == 'optimal_solution':
            # change the settings in order for the online solver to find the actual optimal solution (if possible)
            max_cardinality = len(payload[PayloadKeys.REQUESTS])
            config.MAX_CARDINALITY = max_cardinality
            config.LARGEST_TSP = max_cardinality * 2
            config.RTV_TIMEOUT = 3600 # 1 hour
            config.ILP_TIMEOUT = 3600 # 1 hour
            config.SHARE_COST_FACTOR = 5
            console_logger.info(f"Using settings for optimal solution: MAX_CARDINALITY: {config.MAX_CARDINALITY}, LARGEST_TSP: {config.LARGEST_TSP}, RTV_TIMEOUT: {config.RTV_TIMEOUT}, ILP_TIMEOUT: {config.ILP_TIMEOUT}, SHARE_COST_FACTOR: {config.SHARE_COST_FACTOR}")
            on_solver = OnlineRTVSolver(config)
            updated_driver_runs, _ = on_solver.solve_pdptw_rtv(payload)
        elif config.MODE == 'hexaly_solution':
            # gurobi and hexaly have same interface for now
            # updated_driver_runs, _ = GurobiSolver.solve_pdptw(config.SERVER_URL, payload, time_limit=3600, output_dir=config.OUTPUT_DIR, iteration=0, min_truck=False, dwell_pickup=180, dwell_dropoff=60, tt_matrix=None)
            data = PayloadParser.load_input_data("/Users/jw/Desktop/master_thesis/mt_sourcecode/rtv-solver/outputs/debug/run_20260226_171939_ef1f99/result_driver_runs.json")
            updated_driver_runs, _ = HexalySolver.check_solution(payload, config.SERVER_URL, time_limit=3600, output_dir=config.OUTPUT_DIR)
            # updated_driver_runs, _ = HexalySolver.solve_pdptw(config.SERVER_URL, payload, time_limit=3600, output_dir=config.OUTPUT_DIR, iteration=0, min_truck=False, dwell_pickup=180, dwell_dropoff=60, tt_matrix=None)
        else:
            updated_driver_runs = []
            raise ValueError('No solution as no correct config.MODE provided.')

        # calculate statistics of each iteration; for now only the first vehicle
        tt_matrix = payload.get(PayloadKeys.TIME_MATRIX, None)
        stats_payload = {
            PayloadKeys.DEPOT: payload[PayloadKeys.DEPOT],
            PayloadKeys.REQUESTS: payload[PayloadKeys.REQUESTS],
            PayloadKeys.DRIVERS: updated_driver_runs,
            PayloadKeys.TIME_MATRIX: tt_matrix
        }
        
        stats_evaluator = StatsParser(config, payload = stats_payload)
        total_time = time.time() - start_time
        feasible, stats, violations = stats_evaluator.evaluate(stats_payload)
        stats_evaluator.add_total_time(total_time = total_time)
        assignment_history = stats_evaluator.evaluate_development(stats_payload)
        
        console_logger.info(stats)
        console_logger.info(f'Violations: {violations}')
        console_logger.info(f"Total time: {stats.total_time:.2f}s")

        # console_logger.info("Request history analysed.")
        # console_logger.info(assignment_history)

        save_json(stats_payload, 
                config.OUTPUT_DIR / "result_driver_runs.json")
        save_json({"stats": stats, "violations": violations},
                config.OUTPUT_DIR / "results.json")

        console_logger.info(f"Run complete. Results can be found @ {Path(config.OUTPUT_DIR)}")
    
        # VISUALISE
        if stats.serviced > 0:
            with open(config.OUTPUT_DIR / "result_driver_runs.json", 'r') as driver_runs_file:
                loaded_data = json.load(driver_runs_file)

            if config.SERVER_URL is not None:
                mapper = RouteManifestMapper(config, tt_matrix = tt_matrix)
                geojson = mapper.manifest_to_geojson(loaded_data, 18)
                mapper.save_geojson(geojson, config.OUTPUT_DIR / "route_manifest_v2.geojson")
            else:
                pass # visualisation only possible with server
            

    # not required always, can be easily recreated and is not based on the result but rather the initial data
    # plot_requests_operating_area(payload, show=False, save_path=config.OUTPUT_DIR / "request_distribution.png") 
    
