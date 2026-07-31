import argparse
import logging
import time
import json
import copy

from pathlib import Path

from rtv_solver import OnlineRTVSolver, OfflineRTVSolver, COAMLPipeline # HexalySolver
from rtv_solver.training_loop import COAMLTrainingLoop

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser

from rtv_solver.schema.payload_keys import PayloadKeys

from rtv_solver.structure.config import Config

from rtv_solver.util.logger import setup_loggers, BASIC_LOGGER, DATA_LOGGER
from rtv_solver.util.helper import save_json, set_seed

from rtv_solver.visuals.route_manifest_mapper import RouteManifestMapper

from rtv_solver.pipeline import TYPE_BEST_ORDERED_MATCH
from rtv_solver.pipeline.request_graph_pruner import RequestGraphPruner

if __name__ == "__main__":
    """
    Main script to run RTV solver in online or offline mode based on input data and configuration parameters with easy options for debugging and testing.
    
    Assumptions (wilson - 02.02.2026):
    - All vehicles start from the same location (depot) and need to return there at the end of their shift.
    - Accepted waiting times are already defined in the request payload and currently not defined by the program (wilson always has 30 min with partial overlap - travel_time defines allowed dropoff times; LiLim is much more flexible)
    - The first accepted request of a vehicle is directly considered as boarded (rationale: one vehicle has to commit trip and thus it is already fixed with that vehicle, second request is only considered boarded.)
    - Rebalancing: Rejected requests are in underserved areas, so we need to send additional vehicles there. (not used in current setup)
    """
    # TODO move config management to YAML config or Hydra (no prio)
    parser = argparse.ArgumentParser(description='Arguments for the RTV solver main script')
    # reproduction setup
    parser.add_argument('--config_file', type=str,          default="", help='Path to JSON config file instead of defining the other values manually')
    parser.add_argument('--override', action='append',      default=[], help='Override config values when loading from a config file, e.g. key=value (can be repeated)')
    
    # technical parameters for parallelization
    parser.add_argument('--server_url', type=str,           default="http://127.0.0.1:5001/", help='Backend server URL')
    parser.add_argument('--max_thread_cnt', type=int,       default=16, help='Maximum thread count for parallel processing')
    parser.add_argument('--rtv_timeout', type=int,          default=1800, help='RTV construction timeout in seconds')
    parser.add_argument('--ilp_timeout', type=int,          default=120, help='ILP solver timeout in seconds')
    parser.add_argument('--ilp_penalty', type=int,          default=100_000, help='Penalty for not serving a trip')

    # stats parameters
    parser.add_argument('--travel_time_margin', type=int,   default=5, help='Error margin for travel time in stats calculation')
    
    # experiment parameters (basically default values as they are but can be used)
    parser.add_argument('--debug', type= str, default='False', choices=['True', 'False'], help='Run in debug mode (# reduces number of vehicles and requests for easier debugging)')
    parser.add_argument('--rebalancing', type=str,          default='False', choices=['True', 'False'], help='(NOT WOKRING 12.02.2026) Vehicles are rebalanced if the need arises based on missed requests and idling vehicles.')
    parser.add_argument('--keep_active', type=str,          default='False', choices=['True', 'False'], help='Active requests from an ILP solution in a prior iteration must be kept.')
    parser.add_argument('--return_depot', type=str,         default='True', choices=['True', 'False'], help="Vehicles must return to the originating depot.")
    parser.add_argument('--intermediate_location', type=str, default='False', choices=['True', 'False'], help='Intermediate locations are considered for the solver.')
    parser.add_argument('--dwell_pickup', type=int,         default=180, help='Dwell time at pickup in seconds (backup defaults)')
    parser.add_argument('--dwell_alight', type=int,         default=90, help='Dwell time at alight (dropoff) in seconds (backup defaults)')
    parser.add_argument('--share_cost_factor', type=int,    default=5, help='Shareable cost factor in factor of original single cost')
    parser.add_argument('--walk_distance_cutoff', type=int, default=0, help="Walking distance between dropoff and final destination.")
    
    # RTV generation and rolling horizon parameters (large influence)
    parser.add_argument('--mode', '-m', type=str, choices=['online', 'offline', 'coaml', 'optimal_solution'], default='coaml', help='Mode on how the programme should solve the PDPTW') # TODO fix hexaly "optimal" solution calculator
    parser.add_argument('--max_cardinality', type=int,      default=3, help='Maximum trips to be shared when creating trips in one batch_interval')
    parser.add_argument('--largest_tsp', type=int,          default=16, help='Largest TSP to be solved when constructing RTVs') # incl existing passengers
    parser.add_argument('--step_size', type=int,            default=100, help='Step size in seconds for rolling horizon')
    parser.add_argument('--batch_interval', type=int,       default=400, help='Batch interval in seconds - NOTE if batch_interval value is too small, we might miss requests if the vehicle_trip to the pickup is longer than the batch interval size') # (TODO fix this so this does not have as much impact)
    # NOTE time spans for requests and trips is very different across liLim and wilson, so we need to adjust the values accordingly
    
    # COAML PARAMETERS
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') 
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs over the same payload for COAML mode')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the ML model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for the MLP model')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples for the Fenchel-Young loss')
    parser.add_argument('--sigma', type=float, default=0.2, help='Sigma for the Fenchel-Young loss')
    # 2026-07-30: additive alongside the default ScoringMLP - lets a run use
    # CandidateScoringGNN (rtv_solver/pipeline/candidate_scoring_gnn.py)
    # instead, so both can be trained/evaluated side by side for comparison.
    parser.add_argument('--model_type', type=str, choices=['mlp', 'gnn'], default='mlp', help='Trip-scoring model: "mlp" (ScoringMLP, default) or "gnn" (CandidateScoringGNN, conflict-graph message passing).')
    parser.add_argument('--gnn_num_message_passing_layers', type=int, default=1, help='Only used when --model_type gnn. Number of conflict-graph message-passing layers.')
    parser.add_argument('--y_star_type', type=str, choices=[TYPE_BEST_ORDERED_MATCH], default=TYPE_BEST_ORDERED_MATCH, help='Type of y_star to be used for the Fenchel-Young loss during imitation learning')
    # 2026-07-19: single-file, single-epoch coaml runs (--epochs 1, the "else" branch in
    # the MODE=='coaml' dispatch below) never loaded a pretrained trip-scoring checkpoint
    # and always solved with the default mode="train" (expert-imitation replay) - fine for
    # ad-hoc debugging, but meant there was no way to evaluate an already-trained model's
    # own policy on a fresh holdout file. These two flags close that gap.
    parser.add_argument('--coaml_model_weights', type=str, default="", help='COAML mode, --epochs 1 only: path to a saved trip-scoring checkpoint (from COAMLTrainingLoop/save_model_weights) to load before solving. Empty = fresh untrained model (old default behavior).')
    parser.add_argument('--coaml_solve_mode', type=str, choices=['train', 'eval', 'optimal', 'offline'], default='train', help='COAML mode, --epochs 1 only: which assignment result solve_pdptw uses per iteration - "train" replays expert y*-matching (old default), "eval" uses the model\'s own learned scores, "offline" is pure cost-minimization (same objective as RHO).')

    # FILE MANAGEMENT
    parser.add_argument('--input_file', type=str,           default="solutions/li_lim/manifests/lc101.json", help='Path to a single input file (used for offline mode and coaml single-file runs)')
    parser.add_argument('--input_dir', type=str,            default="solutions/li_lim/manifests/", help='directory of training files for COAML mode')
    parser.add_argument('--val_input_file', type=str,       default="", help='COAML mode, single-payload NYC-style: path to a separate validation manifest. If set (and --input_dir ""), trains on --input_file across --epochs and evaluates on this file each epoch (mode="eval"), saving the best-val-service-rate checkpoint.')
    parser.add_argument('--imitation_solution_file', type=str, default='solutions/li_lim/manifests/lc101.json', help='Path to the imitation solution file with the complete manifest of all trips for all vehicles')
    parser.add_argument('--output_dir', type=str,           default="debug", help='Output directory')
    parser.add_argument('--extra_validation_files', type=str, default="", help='Comma-separated instance stems to ADD to the standard VALIDATION_FILES for this run only (e.g. "lc207,lc208,lr210,lr211,lrc207,lrc208"). Does not change TRAINING_FILES or affect other runs.')
    parser.add_argument('--extra_training_files', type=str, default="", help='Comma-separated instance stems to ADD to the standard TRAINING_FILES for this run only (e.g. "lc201,lc202,...,lrc205"). Actually changes what the SIL scoring MLP trains on - unlike --extra_validation_files, this produces a genuinely different trained model.')
    # 2026-07-26: unlike extra_training_files/extra_validation_files (additive, union
    # with the standard 23/6-file split), these REPLACE the standard split entirely -
    # needed for a "class-2-only" training variant (no way to express "class-1 files
    # minus themselves" via a purely additive flag). Empty (default) = no override,
    # falls back to the standard TRAINING_FILES/VALIDATION_FILES + extra_* as before.
    parser.add_argument('--override_training_files', type=str, default="", help='Comma-separated instance stems that REPLACE (not add to) the standard TRAINING_FILES for this run only (e.g. for a class-2-only variant: "lc201,lc202,lc203,lc205,lc206,lr201,...").')
    parser.add_argument('--override_validation_files', type=str, default="", help='Comma-separated instance stems that REPLACE (not add to) the standard VALIDATION_FILES for this run only (e.g. "lc207,lc208,lr210,lr211,lrc207,lrc208").')

    # Request graph pruning parameters
    parser.add_argument(
        "--use_request_graph_pruner",
        type=str,
        default="False",
        help="Enable GNN-based request graph pruning before shared trip generation."
    )

    parser.add_argument(
        "--request_graph_model_path",
        type=str,
        # 2026-07-07: default switched to the best v2 GNN checkpoint (pw10, thr=0.5);
        # the old path was a v1-shaped checkpoint incompatible with RequestGraphEdgeGNNv2.
        default="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt",
        help="Path to the trained request graph GNN model."
    )

    parser.add_argument(
        "--request_graph_threshold",
        type=float,
        default=0.5,
        help="Score threshold used to keep request-request pairs."
)

    # 2026-07-14: Request pruning parameters - request-only pruner
    # (RequestPruner), separate from and runs before the request-graph pair
    # pruner above. See rtv_solver/pipeline/request_pruner.py.
    parser.add_argument(
        "--use_request_pruner",
        type=str,
        default="False",
        help="Enable MLP-based request-only pruning before trip generation."
    )
    # parser for new imitation scoring rule (26.7)
    parser.add_argument(
        "--imitation_scoring_rule",
        type=str,
        choices=["legacy", "exponential_prefix"],
        default="legacy"
    )
    parser.add_argument(
        "--request_pruner_model_path",
        type=str,
        default="outputs/request_pruner_mlp/request_pruner_mlp_h32_l1_d0p0_pw1p0/request_pruner_mlp_h32_l1_d0p0_pw1p0_best_val_f3.pt",
        help="Path to the trained request pruner MLP checkpoint."
    )

    parser.add_argument(
        "--request_pruner_threshold",
        type=float,
        default=0.3,
        help="Score threshold used to keep individual requests (candidates only - urgent requests are always force-kept, see RequestPruner)."
    )
    # alternative input files
    # parser.add_argument('--input_file', type=str,           default="test_nc/ttm/test_10r_1v_repeat6_simple.pkl", help='Request input file') 
    # parser.add_argument('--input_file', type=str,           default="inputs/wilson_nc_initial.pkl", help='Request input file') 
    # TODO fix input handling for Sartori datasets (result does not work as expected and has many violations) - no priority to fix this at end of thesis
    # parser.add_argument('--input_file', type=str,           default="sartori/n100/bar-n100-1.txt")
    # TODO fix input handling for chattanooga datasets (violations after offline run) - no priority to fix this at end of thesis
    # parser.add_argument('--input_file', type=str,           default="inputs/localDB_payload_oct.pkl")
    

    # TODO it should be possible to implement multiple NN in parallel while running the COAML pipeline as during training we follow the optimal solution in steps. it should be analysed whether with the rolling horizon how close we actually get to the optimal solution with taking the optimal steps
    # however, one can train multiple NN and then run multiple validation in the end with the different decision-making, but overall it should parallelize the training process and experimentation; requires a few changes in the infrastructure code base and it would probably help if one can reload the model to just run validation with the different models at the very end (that could be one of the main advantages as we have a fast CO layer.)
    
    # TODO add a tracker for the runtime to compare the speed both for training and the validation, is probably going to be quite interesting

    # TODO one needs a deterministic and quick check whether the current rolling horizon parameters are actually able to rebuild the optimal solution and whether the real trip can be rebuilt from it. this should not be done via running the COAML pipeline as it takes too much time. IDEA: check only for the runs that one needs if they can be created for the optimal solution and simulate them accordingly. RTV might still break this as the order of the pickups and dropoffs is not guaranteed to be the same; there must be a way and that also helps to argue about the rolling horizon parameters.

    # TODO ask Samitha again for the paratransit-adjusted dataset as this should work much better than the original work

    # implement configurations
    arguments = parser.parse_args()
    config = Config.from_args(arguments)
    config.enforce_constraints()
    set_seed(config.SEED, config.DEBUG)

    setup_loggers(config.OUTPUT_DIR)
    console_logger = logging.getLogger(BASIC_LOGGER)
    data_logger = logging.getLogger(DATA_LOGGER)

    console_logger.info(f"Input file: {config.INPUT_FILE}")
    if config.INPUT_DIR:
        console_logger.info(f"Input dir (train): {config.INPUT_DIR}")
    console_logger.info(f"Output directory: {config.OUTPUT_DIR}")
    console_logger.info(f' --- Start: RTV simulation --- mode > {config.MODE}')
    console_logger.info(f'Arguments: {config}')

    # COAML train/val dirs: training loop handles everything (load, train, validate, store)
    if config.MODE == 'coaml' and config.INPUT_DIR:
        extra_val_files = [s.strip() for s in config.EXTRA_VALIDATION_FILES.split(",") if s.strip()]
        extra_train_files = [s.strip() for s in config.EXTRA_TRAINING_FILES.split(",") if s.strip()]
        override_train_files = [s.strip() for s in config.OVERRIDE_TRAINING_FILES.split(",") if s.strip()]
        override_val_files = [s.strip() for s in config.OVERRIDE_VALIDATION_FILES.split(",") if s.strip()]
        training_loop = COAMLTrainingLoop(
            config,
            extra_validation_files=extra_val_files,
            extra_training_files=extra_train_files,
            override_training_files=override_train_files,
            override_validation_files=override_val_files,
        )
        training_result = training_loop.run()
        console_logger.info(f"All iteration losses: {training_result.all_iteration_losses}")
        console_logger.info(f"Run complete. Results @ {Path(config.OUTPUT_DIR)}")
    else: # single file_handling
        # load data from file and update to canonical format for the entire system
        data = PayloadParser.load_input_data(Path(__file__).resolve().parent.parent / config.INPUT_FILE)

        if config.DEBUG: # check if the basic functionality of the online RTV solver works (foundation for offline RTV solver)
            console_logger.setLevel(logging.INFO)
            config.RTV_TIMEOUT = 600000 # clicking through inputs, it should not break due to timeout
            # reduce the complexity by only considering a small number of vehicles and requests
            driver_runs_total = data[PayloadKeys.DRIVERS]
            driver_runs_reduced = driver_runs_total[:5]
            # test to change the first vehicle to trigger certain situations
            vehicle_state = driver_runs_reduced[0][PayloadKeys.DRIVER_STATE]
            vehicle_manifest = driver_runs_reduced[0][PayloadKeys.DRIVER_MANIFEST]
            # COUNT_BASED SELECTION OF REQUESTS: add code that just takes the first n requests out of the payload
            fixed_data = copy.deepcopy(data)
            selected_requests = fixed_data[PayloadKeys.REQUESTS][:20]
            payload = {
                PayloadKeys.DEPOT: data[PayloadKeys.DEPOT],
                PayloadKeys.REQUESTS: selected_requests,
                PayloadKeys.DRIVERS: driver_runs_reduced}
        else:
            payload = data

        # required to run data from other sources without the backend server
        if data.get(PayloadKeys.TIME_MATRIX) is not None:
            payload[PayloadKeys.TIME_MATRIX] = data[PayloadKeys.TIME_MATRIX]
        else:
            payload[PayloadKeys.TIME_MATRIX] = None
            console_logger.warning("Time matrix is not available. Solution run on server, but time_matrix is missing - leading to no possibility of running this dataset without backend server.")

        # Initialize RTV solver
        start_time = time.time()
        if config.MODE == 'online':
            on_solver = OnlineRTVSolver(config)
            updated_driver_runs, _ = on_solver.solve_pdptw_rtv(payload)
        elif config.MODE == 'offline':
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            off_solver = OfflineRTVSolver(config)
            updated_driver_runs = off_solver.solve_rtv(cleared_payload, config.BATCH_INTERVAL, config.STEP_SIZE)
        elif config.MODE == 'coaml':
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            input_path = Path(__file__).resolve().parent.parent / config.INPUT_FILE
            if config.EPOCHS > 1:
                # 2026-07-19: NYC-style single-payload train/val - separate from the
                # Li&Lim --input_dir path (many small instance files). Trains the
                # trip-scoring model on cleared_payload across config.EPOCHS, evaluating
                # (mode="eval", no gradient) on VAL_INPUT_FILE each epoch and keeping the
                # best-val-service-rate checkpoint - see
                # COAMLTrainingLoop._run_train_val_payloads for why this was needed
                # (previously every NYC "SIL" run trained AND was scored on the exact
                # same small holdout payload, mode="train" only, no real val/eval split).
                val_payload = None
                val_input_path = None
                if config.VAL_INPUT_FILE:
                    val_input_path = Path(__file__).resolve().parent.parent / config.VAL_INPUT_FILE
                    val_data = PayloadParser.load_input_data(val_input_path)
                    val_payload = PayloadParser.clear_vehicle_manifests(val_data)
                    val_payload[PayloadKeys.TIME_MATRIX] = val_data.get(PayloadKeys.TIME_MATRIX)
                training_loop = COAMLTrainingLoop(
                    config, cleared_payload, input_path=input_path,
                    val_payload=val_payload, val_input_path=val_input_path,
                )
                training_result = training_loop.run()
                updated_driver_runs = training_result.updated_driver_runs
                console_logger.info(f"All iteration losses: {training_result.all_iteration_losses}")
            else:
                rh_solver = COAMLPipeline(
                    config, cleared_payload, imitation_solution_path=input_path
                )
                if config.COAML_MODEL_WEIGHTS:
                    rh_solver.load_model_weights(config.COAML_MODEL_WEIGHTS)
                updated_driver_runs = rh_solver.solve_pdptw(cleared_payload, mode=config.COAML_SOLVE_MODE)
                console_logger.info(f"Loss history: {rh_solver.loss_history}")
        elif config.MODE == 'optimal_solution':
            max_cardinality = len(payload[PayloadKeys.REQUESTS])
            config.MAX_CARDINALITY = max_cardinality
            config.LARGEST_TSP = max_cardinality * 2
            config.RTV_TIMEOUT = 3600
            config.ILP_TIMEOUT = 3600
            config.SHARE_COST_FACTOR = 5
            console_logger.info(f"Using settings for optimal solution: MAX_CARDINALITY: {config.MAX_CARDINALITY}, LARGEST_TSP: {config.LARGEST_TSP}, RTV_TIMEOUT: {config.RTV_TIMEOUT}, ILP_TIMEOUT: {config.ILP_TIMEOUT}, SHARE_COST_FACTOR: {config.SHARE_COST_FACTOR}")
            on_solver = OnlineRTVSolver(config)
            updated_driver_runs, _ = on_solver.solve_pdptw_rtv(payload)
        elif config.MODE == 'hexaly_solution':
            updated_driver_runs = []
            pass  # not implemented
            # gurobi and hexaly have same interface for now
            # updated_driver_runs, _ = GurobiSolver.solve_pdptw(config.SERVER_URL, payload, time_limit=3600, output_dir=config.OUTPUT_DIR, iteration=0, min_truck=False, dwell_pickup=180, dwell_dropoff=60, tt_matrix=None)
            # data = PayloadParser.load_input_data("/Users/jw/Desktop/master_thesis/mt_sourcecode/rtv-solver/outputs/debug/run_20260226_171939_ef1f99/result_driver_runs.json")
            # updated_driver_runs, _ = HexalySolver.check_solution(payload, config.SERVER_URL, time_limit=3600, output_dir=config.OUTPUT_DIR)
            # updated_driver_runs, _ = HexalySolver.solve_pdptw(config.SERVER_URL, payload, time_limit=3600, output_dir=config.OUTPUT_DIR, iteration=0, min_truck=False, dwell_pickup=180, dwell_dropoff=60, tt_matrix=None)
    
        else:
            updated_driver_runs = []
            raise ValueError('No solution as no correct config.MODE provided.')

        # calculate statistics and save
        tt_matrix = payload.get(PayloadKeys.TIME_MATRIX, None)
        stats_payload = {
            PayloadKeys.DEPOT: payload[PayloadKeys.DEPOT],
            PayloadKeys.REQUESTS: payload[PayloadKeys.REQUESTS],
            PayloadKeys.DRIVERS: updated_driver_runs,
            PayloadKeys.TIME_MATRIX: tt_matrix
        }
        stats_evaluator = StatsParser(config, payload=stats_payload)
        total_time = time.time() - start_time
        feasible, stats, violations = stats_evaluator.evaluate(stats_payload)
        stats_evaluator.add_total_time(total_time=total_time)
        assignment_history = stats_evaluator.evaluate_development(stats_payload)

        console_logger.info(stats)
        console_logger.info(f'Violations: {violations}')
        console_logger.info(f"Total time: {stats.total_time:.2f}s")
        console_logger.info(stats.format_vehicle_leg_report())

        final_output_dir = config.OUTPUT_DIR / "final"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        save_json(stats_payload, final_output_dir / "result_driver_runs.json")
        save_json({"stats": stats, "violations": violations}, final_output_dir / "results.json")

        console_logger.info(f"Run complete. Results can be found @ {Path(config.OUTPUT_DIR)}")

    # VISUALISE NOTE (not run during cluster runs)
    # if stats.serviced > 0:
    #     with open(config.OUTPUT_DIR / "result_driver_runs.json", 'r') as driver_runs_file:
    #         loaded_data = json.load(driver_runs_file)

    #     if config.SERVER_URL is not None:
    #         mapper = RouteManifestMapper(config, tt_matrix = tt_matrix)
    #         geojson = mapper.manifest_to_geojson(loaded_data, 18)
    #         mapper.save_geojson(geojson, config.OUTPUT_DIR / "route_manifest_v2.geojson")
    #     else:
    #         pass # visualisation only possible with server

    # not required always, can be easily recreated and is not based on the result but rather the initial data
    # plot_requests_operating_area(payload, show=False, save_path=config.OUTPUT_DIR / "request_distribution.png") 
