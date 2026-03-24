import multiprocessing
import sys
from multiprocessing import Pool
import numpy as np
import copy
from typing import Any, Optional
from pathlib import Path
import torch

from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.online_rtv_solver import OnlineRTVSolver

from rtv_solver.schema.payload_keys import PayloadKeys

from rtv_solver.structure.config import Config
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.assignment_result import AssignmentResult

from rtv_solver.pipeline import CO, CO_ScoreMaximization, CO_TripCostMinimization, CO_RebalancingCoverage, FeatureBuilder
from rtv_solver.pipeline import FenchelYoungLoss, make_map_oracle, ScoringMLP
from rtv_solver.pipeline.imitation_handler import ImitationHandler, TYPE_BEST_ORDERED_MATCH, TYPE_BEST_UNORDERED_MATCH

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class COAMLPipeline():
    """
    Implements COAML pipeline for a rolling-horizon solution for the PDPTW.
    
    Some functions in the OnlineRTVSolver can be reused to check feasibility.

    When a scoring model and Fenchel-Young loss are supplied, each call to
    solve_iteration() additionally computes a training loss and stores it in
    self.last_loss.  The vehicle assignment itself is always determined by
    CO_TripCostMinimization (unchanged), so routing quality is unaffected.

    Training loop usage (not implemented here)::

        for t, payload in stream:
            optimizer.zero_grad()
            driver_runs = pipeline.solve_iteration(payload, t)
            if pipeline.last_loss is not None:
                pipeline.last_loss.backward()
                optimizer.step()
    """
    def __init__(
            self,
            config: Config,
            offline_payload: dict,
            model: ScoringMLP | None = None,
            epoch: int = 1,
            imitation_solution_path: Path | str | None = None,
        ):
        """
        Initialize the COAML pipeline solver.

        Parameters:
            - config: Config object
            - offline_payload: Offline payload object in order to calculate feature normalization values and get structure of feature matrix
            - imitation_solution_path: Path to the file containing the optimal solution (same file the pipeline runs on). If None, ImitationHandler falls back to config.IMITATION_SOLUTION_FILE.
        """
        self.config = config
        self.offline_payload = offline_payload
        self.feature_builder = FeatureBuilder(offline_payload, self.config)
        self.model = model if model is not None else ScoringMLP(
            feature_dim=FeatureBuilder.FEATURE_SIZE, hidden_dim=config.HIDDEN_DIM
        )
        self.fy_loss = FenchelYoungLoss(num_samples=config.NUM_SAMPLES, sigma=config.SIGMA) # alternative 0.1 or 0.05 for less variance
        self.imitation_handler = ImitationHandler(config, imitation_solution_path=imitation_solution_path)
        
        self.coaml_optimizer = CO_ScoreMaximization(config)
        self.default_optimizer = CO_TripCostMinimization(config)    
        
        self.last_loss: Optional[torch.Tensor] = None
        self.loss_history: list[Optional[float]] = []
        self.epoch = epoch

        if sys.platform == "darwin": # required to run correctly on MacOS
            try:
                # it is required here as we do not call OnlineRTVSolver as an object
                multiprocessing.set_start_method("fork")
                # NOTE if you see issues like this: 'Fatal Python error: Bus error' you might need to introduce spawn context for each Multiprocessing Pool (also depends on the cluster system)
            except RuntimeError: # start method was already set somewhere else -> don't crash
                pass

    def solve_pdptw(
        self,
        payload: dict,
        mode: str = "train",
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        # TODO there is probably a better way of coding to run the same loop for offline and COAML instead of this code duplication that we currently have with all its problems (no thesis prio)

        Handles payloads across iterations of the rolling horizon
        
        With config.return_depot, this method will not add the final trips to the depot. The user has to call finalize_driverRuns(...) to add the final stops.
        """
        # determine time interval of entire requests set
        self.loss_history = []
        start_time, end_time = PayloadParser.get_requests_time_interval(payload)
        # start before the initial start_time to catch all requests in the first interval
        current_time = max(0, start_time - self.config.BATCH_INTERVAL)

        # track progress of solver iterations
        iteration = 0

        driver_runs = payload[PayloadKeys.DRIVERS]

        while current_time < end_time:
            console_logger.info(f"=== Iteration {self.epoch}-{iteration} COAML Solver (mode: {mode}) at time {current_time} ===")
            
            # select requests that are to be considered in the current interval with pickup_window [current_time, current_time + interval]
            # TODO check which definition we should use for the request selection
            # old one: request[PayloadKeys.REQ_PICKUP_WINDOW_START] > current_time and request[PayloadKeys.REQ_PICKUP_WINDOW_START] < current_time + self.config.BATCH_INTERVAL):
                    
            selected_requests = {}
            for request in payload[PayloadKeys.REQUESTS]:
                # pickup window overlaps [current_time, current_time + batch_interval)
                if (request[PayloadKeys.REQ_PICKUP_WINDOW_END] > current_time and
                    request[PayloadKeys.REQ_PICKUP_WINDOW_START] < current_time + self.config.BATCH_INTERVAL):
                    selected_requests[request[PayloadKeys.REQ_BOOKING_ID]] = request
            
            # remove requests that are already part of vehicles or have already been dropped off; covered by PayloadParser in OnlineRTVsolver
            for dr in driver_runs:
                manifest = dr[PayloadKeys.DRIVER_MANIFEST]
                for stop in manifest:
                    if stop[PayloadKeys.MANIFEST_BOOKING_ID] in selected_requests:
                        del selected_requests[stop[PayloadKeys.MANIFEST_BOOKING_ID]]
            selected_requests = list(selected_requests.values())

            # create a new payload with the selected requests
            new_payload = {
                PayloadKeys.DEPOT: payload[PayloadKeys.DEPOT],
                PayloadKeys.REQUESTS: selected_requests,
                PayloadKeys.DRIVERS: driver_runs,
                PayloadKeys.CURRENT_TIME: current_time,
                PayloadKeys.TIME_MATRIX: payload.get(PayloadKeys.TIME_MATRIX, None)}

            # solve the RTV problem and update manifests
            if len(selected_requests) == 0:
                new_driver_runs = driver_runs
            else:    
                new_driver_runs = self.solve_iteration(new_payload, iteration, mode = mode)
                if mode == "train" and optimizer is not None and self.last_loss is not None:
                    optimizer.zero_grad(set_to_none=True)
                    self.last_loss.backward()
                    optimizer.step()
                          
            # increment time (might not be the size of the batch) and iteration
            current_time += self.config.STEP_SIZE 
            iteration += 1

            # update vehicles based on decisions in the previous step until current time (might not be the entire interval)
            simulated_driver_runs = OnlineRTVSolver.simulate_manifest(self.config, current_time, new_driver_runs, tt_matrix=new_payload[PayloadKeys.TIME_MATRIX])
            driver_runs = simulated_driver_runs

        final_driver_runs = OnlineRTVSolver.finalize_driverRuns(
            self.config, driver_runs, payload[PayloadKeys.DEPOT]
        )
        # self.save_model_weights()
        return final_driver_runs

    def _default_model_weights_path(self) -> Path:
        """
        Default checkpoint path for this run.
        """
        return Path(self.config.OUTPUT_DIR) / "coaml_model_weights.pt"

    def save_model_weights(self, path: str | Path | None = None) -> Path:
        """
        Save neural-network weights so training can be resumed later.

        The checkpoint contains the model state dict and lightweight metadata
        about model dimensions/features used during this run.
        """
        checkpoint_path = Path(path) if path is not None else self._default_model_weights_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "feature_dim": self.model.feature_dim,
            "hidden_dim": self.model.hidden_dim,
            "feature_size": FeatureBuilder.FEATURE_SIZE,
        }
        torch.save(checkpoint, checkpoint_path)
        console_logger.info(f"Saved COAML model weights to {checkpoint_path}")
        return checkpoint_path

    def load_model_weights(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> None:
        """
        Load neural-network weights from a checkpoint file.

        Use this to continue training on new data in later runs.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=strict)
        console_logger.info(f"Loaded COAML model weights from {checkpoint_path}")

    def _build_imitation_scores_with_reject(
        self,
        trip_costs: list[TripCost],
        request_batch: list,
        reject_vehicle_ids: list[int] | None = None,
        vehicle_to_trips_cost_map: dict[int, list[int]] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Build vehicle-aware imitation scores and append reject-action scores.

        Returns:
            - imitation_scores_with_reject: score vector with layout `[trip_scores..., reject_scores...]`
            - trip_vehicle_ids: vehicle id per trip score entry, aligned with the first `len(trip_costs)` indices
        """
        trip_scores, trip_vehicle_ids = self.imitation_handler.score_trip_costs_against_optimal_by_vehicle(
            trip_costs=trip_costs,
            request_batch=request_batch,
            vehicle_to_trips_cost_map=vehicle_to_trips_cost_map,
        )
        imitation_scores_with_reject = self.imitation_handler.append_reject_action_scores(
            trip_scores,
            reject_vehicle_ids or [],
            trip_vehicle_ids=trip_vehicle_ids,
        )
        if trip_scores.shape[0] != len(trip_costs):
            raise ValueError(
                "Trip imitation-score length mismatch: "
                f"got {trip_scores.shape[0]}, expected {len(trip_costs)}."
            )
        expected_total = len(trip_costs) + len(reject_vehicle_ids or [])
        if imitation_scores_with_reject.shape[0] != expected_total:
            raise ValueError(
                "Imitation-score vector length mismatch after reject append: "
                f"got {imitation_scores_with_reject.shape[0]}, expected {expected_total}."
            )
        return imitation_scores_with_reject, trip_vehicle_ids

    def _build_y_star_from_imitation_scores(
        self,
        imitation_scores: torch.Tensor,
        trip_vehicle_ids: list[int],
        reject_vehicle_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Convert imitation scores into binary y* according to configured strategy.
        """
        reject_vehicle_ids = reject_vehicle_ids or []
        if self.config.Y_STAR_TYPE == TYPE_BEST_ORDERED_MATCH:
            y_star = ImitationHandler.build_y_star_per_vehicle_from_imit_scores(
                imitation_scores_with_reject=imitation_scores,
                trip_vehicle_ids=trip_vehicle_ids,
                reject_vehicle_ids=reject_vehicle_ids,
            )
            if y_star.shape[0] != imitation_scores.shape[0]:
                raise ValueError(f"y_star shape mismatch: got {y_star.shape[0]}, expected {imitation_scores.shape[0]}.")
            if not torch.all((y_star == 0) | (y_star == 1)):
                raise ValueError("y_star must be binary (0/1).")
            return y_star
        raise ValueError(f"Invalid y_star type: {self.config.Y_STAR_TYPE}")
    
    def solve_iteration(self, subset_payload, iteration = 0, mode: str = "train"):
        """
        Solver for the entire payload that is given, based on the onlineRTVSolver but adapted to COAML pipeline.
        """
        # TODO (major effort) improve code quality as we currently have a lot of repetition that should not be required as we need similar information in online, offline and COAML
        
        # initialize network and payload
        needs_server_matrix_build = NetworkHandler.init_from_payload(
            payload=subset_payload,
            server_url=self.config.SERVER_URL,
        )
        
        payload_object = PayloadParser.get_payload_object(subset_payload, dwell_pickup_default=self.config.DWELL_PICKUP, dwell_alight_default=self.config.DWELL_ALIGHT, online=False)
        request_handler = RequestHandler(payload_object.requests, config=self.config)
        request_batch, active_requests, boarded_requests = request_handler.get_request_batches(payload_object)
        vehicle_handler = VehicleHandler(payload_object.depot, 
                                         payload_object.driver_runs,
                                         self.config)
        
        
        # create trips of all already boarded requests 
        # NOTE these requests are not always boarded at this point but might be still committed to a vehicle (especially if it is the first request of an idling vehicle)
        boarded_trips = TripHandler.create_trip_for_picked_requests(boarded_requests, iteration)
        # update vehicle position/trips/times along its path according to all data stored in the manifest
        vehicle_handler.add_manifest_to_vehicles(payload_object.driver_runs,
                                                 boarded_requests,
                                                 boarded_trips)
        if needs_server_matrix_build:
            NetworkHandler.initialize_travel_time_matrix()
        iteration += 1  # increase iteration as the prior step was just rebuilding from the last iteration (if there was a prior step)
        
        try:
            # generate and assign trips to each vehicle using the RTV approach solved by an ILP
            trip_handler = TripHandler(
                vehicle_handler.vehicles,
                request_batch,
                active_requests,
                iteration,
                self.config)
        except Exception as e:
            raise e
        
        loss_tracked = False

        if len(vehicle_handler.vehicles) != 0:
            single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map = trip_handler.run()
            
            # Split run() so we can capture x_t for y_star extraction; result is identical to self.optimizer.run().
            self.coaml_optimizer.reset(single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map)
            self.default_optimizer.reset(single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map)

            # calculate solution based on 
            feature_matrix, feature_names = self.feature_builder.build_matrix_from_trip_handler(
                trip_handler, payload_object.current_time
            )
            feature_matrix_with_reject = feature_matrix
            reject_vehicle_ids: list[int] = []
            if feature_names:
                # Appends one synthetic reject row per vehicle at the end of the matrix.
                feature_matrix_with_reject, reject_vehicle_ids = self.feature_builder.add_reject_action_entries(
                    feature_matrix,
                    feature_names,
                    vehicle_handler.vehicles,
                    payload_object.current_time,
                )
            feature_tensor_with_reject = torch.tensor(
                feature_matrix_with_reject, dtype=torch.float32
            )
            if len(feature_tensor_with_reject) > 0: # TODO we need to be able to handle the case where there are no reject actions
                # situation: debug_case1.sh returns an empty set of requests, so feature_tensor_with_reject is empty and then we do not have optimal solution or score_solution. Solutoin: either handle it without any features etc. and return the reject action or set up a backup for when the result is selected (because optimaL_result is thus referenced before assignment)
                feature_scores_with_reject = self.model(feature_tensor_with_reject)
                num_reject_actions = len(reject_vehicle_ids)
                if num_reject_actions > 0:
                    # Respect the same ordering as in add_reject_action_entries():
                    # [trip rows..., reject rows...].
                    feature_scores = feature_scores_with_reject[:-num_reject_actions]
                    reject_action_scores = feature_scores_with_reject[-num_reject_actions:]
                else:
                    feature_scores = feature_scores_with_reject
                    reject_action_scores = torch.empty(0, dtype=feature_scores_with_reject.dtype)
                
                ilp_model, x_t, x_r, x_reject = self.coaml_optimizer.solve_ilp(
                    feature_scores,
                    request_batch, 
                    active_requests,
                    penalty=self.config.ILP_PENALTY,
                    keep_active=self.config.KEEP_ACTIVE,
                    reject_action_scores=reject_action_scores.detach().cpu().numpy(),
                    reject_vehicle_ids=reject_vehicle_ids,
                )
                score_result = self.coaml_optimizer.transform_solution_to_assignment(
                    ilp_model,
                    x_t,
                    x_r,
                    request_batch,
                    x_reject=x_reject,
                    reject_vehicle_ids=reject_vehicle_ids,
                )
                imitation_scores_with_reject, trip_vehicle_ids = self._build_imitation_scores_with_reject(
                    trip_costs=trip_costs,
                    request_batch=request_batch,
                    reject_vehicle_ids=reject_vehicle_ids,
                    vehicle_to_trips_cost_map=vehicle_to_trips_cost_map,
                )
                y_star = self._build_y_star_from_imitation_scores(
                    imitation_scores=imitation_scores_with_reject,
                    trip_vehicle_ids=trip_vehicle_ids,
                    reject_vehicle_ids=reject_vehicle_ids,
                )

                # calculate the true optimal solution based on the imitation scores
                optimal_result = self.coaml_optimizer.transform_optimal_solution_to_assignment(
                    y_star,
                    request_batch,
                    reject_vehicle_ids=reject_vehicle_ids,
                )

                console_logger.info(f"Score result: {score_result.request_assignment}, cost: {score_result.added_distance}, rejected: {score_result.unassigned_trip_count}")
                console_logger.info(f"Optimal result: {optimal_result.request_assignment}, cost: {optimal_result.added_distance}, rejected: {optimal_result.unassigned_trip_count}")
            
                # transform solution to assignment, used to update vehicles and manifests
                # THIS result must be used for future comparisons
            
            # calculate default solution based on the ILP minimizing trip costs (used for FY loss), actually not really required as we have the optimal solution available, definitely not in this run because it just reiterates the offline solution
            trip_obj_scores = np.fromiter((tc.cost for tc in trip_costs), dtype=float, count=len(trip_costs))
            default_ilp_model, default_x_t, default_x_r = self.default_optimizer.solve_ilp(
                trip_obj_scores, 
                request_batch, 
                active_requests, 
                penalty=self.config.ILP_PENALTY, 
                keep_active=self.config.KEEP_ACTIVE)
            default_result = self.default_optimizer.transform_solution_to_assignment(
                default_ilp_model,
                default_x_t,
                default_x_r,
                request_batch,
                x_reject=None,
                reject_vehicle_ids=[],
            )
            console_logger.info(f"Default result: {default_result.request_assignment}, cost: {default_result.added_distance}, rejected: {default_result.unassigned_trip_count}")
            console_logger.info(f"Complete solution: {self.imitation_handler.optimal_solution}")
            
            # NOTE this is code to check results more easily from terminal - no priority to keep this code 
            if len(feature_tensor_with_reject) > 0:
                for idx, (ml_score, tc, imitation_score) in enumerate[TripCost](
                    zip(feature_scores, trip_costs, imitation_scores_with_reject[:len(trip_costs)])
                ):  
                    # print each score from the three different result assignments (score, optimal, default)
                    selected_by_score = x_t[idx].X > 0.5
                    selected_by_optimal = y_star[idx].item() > 0.5
                    selected_by_default = default_x_t[idx].X > 0.5
                    if not (selected_by_score or selected_by_optimal or selected_by_default):
                        continue
                    for selected_from, selected in (
                        ("score", selected_by_score),
                        ("optimal", selected_by_optimal),
                        ("default", selected_by_default),
                    ):
                        if not selected:
                            continue
                        print(
                            f"{selected_from} - TC {tc.trip_no} vehicle {tc.vehicle_id} score: {ml_score:3.3f}, "
                            f"imit: {imitation_score.item()}, tc: {tc.get_ordered_request_ids()}"
                        )

                if len(reject_vehicle_ids) > 0:
                    reject_imitation_scores = imitation_scores_with_reject[-len(reject_vehicle_ids):]
                    for idx, vehicle_id in enumerate(reject_vehicle_ids):
                        # Print reject rows only if selected by score or optimal.
                        reject_selected_by_score = (x_reject[vehicle_id].X > 0.5 if vehicle_id in x_reject else False)
                        reject_selected_by_optimal = (y_star[len(trip_costs) + idx].item() > 0.5)
                        if not (reject_selected_by_score or reject_selected_by_optimal):
                            continue
                        reject_selected_from = "|".join(
                            name for name, selected in (
                                ("score", reject_selected_by_score),
                                ("optimal", reject_selected_by_optimal),
                            ) if selected
                        )
                        reject_score = reject_action_scores[idx]
                        reject_imit = reject_imitation_scores[idx]
                        print(
                            f"r{vehicle_id}: from {reject_selected_from}, "
                            f"score: {reject_score}, imit {reject_imit}"
                        )
            
            # train mode, keep on right track - decide which run should move forward
            # NOTE this needs a change when we decide what to do with training or if we want to use the default or score solution from NN
            # TODO for the eval mode in order to accelerate it, it might make sense to turn on config.KEEP_ACTIVE to make it faster?, currently KEEP_ACTIVE does not work in COAML
            if mode == "train": # keep on optimal track
                result = optimal_result 
            elif mode == "eval": # use score solution from NN to decide which run should move forward
                result = score_result
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # compute Fenchel-Young loss from known optimal solution
            if len(feature_tensor_with_reject) > 0:
                self._compute_fy_loss_from_optimal_solution(
                    feature_matrix_with_reject,
                    single_trip_map, 
                    trip_list, 
                    trip_costs,
                    vehicle_to_trips_cost_map, 
                    trip_to_vehicle_cost_map,
                    request_batch, 
                    active_requests,
                    reject_vehicle_ids,
                )
                loss_tracked = True

            if self.config.REBALANCING:
                rebalancing_optimizer = CO_RebalancingCoverage(self.config)
                result = rebalancing_optimizer.run(result, vehicle_handler.vehicles, request_batch)

        if not loss_tracked:
            self.last_loss = None

        if self.last_loss is not None:
            current_loss = float(self.last_loss.detach().cpu().item())
            self.loss_history.append(current_loss)
        else:
            self.loss_history.append(None)
   
        # assign vehicles and add trips / sequence to each vehicle 
        unserved_requests = set([req.id for req in request_batch]) # number of requests that are not already confirmed to be  served
        for vehicle_id in result.vehicle_assignment: # if it is empty the assignment is skipped
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips, prev_sequence = result.vehicle_assignment[vehicle_id]
            plan = VehicleHandler.plan_trip_insertions(vehicle, trips, prev_sequence=prev_sequence)
            vehicle.apply_trip_insertion(plan)
            for trip in trips: # remove assigned trips from unserved
                if trip.request_id in unserved_requests:
                    unserved_requests.remove(trip.request_id)

        # TODO add rebalancing handling based on rebalancing_assignment that already checks required permissions (should not be required for our problem setting)

        # update driver runs
        updated_driver_runs = []
        for driver_run in payload_object.driver_runs:
            new_driver_run = vehicle_handler.update_run(driver_run)
            # new_driver_run = self.update_run(vehicle_handler, driver_run)
            updated_driver_runs.append(new_driver_run)
            
        # check invariants whether manifest is still correct
        OnlineRTVSolver._check_consistency_of_manifests(
            subset_payload[PayloadKeys.DRIVERS], 
            updated_driver_runs,
            unserved_requests, 
            subset_payload[PayloadKeys.REQUESTS],
            keep_active=self.config.KEEP_ACTIVE,
            return_depot=self.config.RETURN_DEPOT)
        
        # collect information on assignment history (especially interesting if config.KEEP_ACTIVE is set to False)
        self._log_assignment_status(result, unserved_requests, payload_object.current_time)
        return updated_driver_runs

    def _compute_fy_loss_from_default_ilp(
        self,
        feature_matrix: np.ndarray,
        ilp_model,
        x_t,
        single_trip_map: dict,
        trip_list: list,
        trip_costs: list,
        vehicle_to_trips_cost_map: dict,
        trip_to_vehicle_cost_map: dict,
        request_batch: list,
        active_requests: dict,
    ) -> None:
        """
        Compute and store the Fenchel-Young loss for the current iteration.

        Uses the CO_TripCostMinimization solution as y_star (ground truth) and CO_ScoreMaximization as the MAP oracle for perturb-and-MAP.

        The result is stored in self.last_loss.  If the ILP did not yield a feasible solution (y_star is all-zero), the loss is skipped and self.last_loss is set to None.

        Side effects:
            Sets self.last_loss.
        """
        y_star = CO.extract_y_binary(ilp_model, x_t) # get optimal solution from default ILP model

        if y_star.sum().item() == 0:
            console_logger.warning("FY loss skipped: Optimizer did not assign any RTVs.")
            self.last_loss = None
            return

        feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        scores = self.model(feature_tensor)    

        oracle = make_map_oracle(
            self.coaml_optimizer,
            single_trip_map,
            trip_list,
            trip_costs,
            vehicle_to_trips_cost_map,
            trip_to_vehicle_cost_map,
            request_batch,
            active_requests,
            self.config,
        )

        self.last_loss = self.fy_loss(scores, y_star, oracle)
        console_logger.info(f"Default FY loss computed: {self.last_loss.item():.4f}")

    def _compute_fy_loss_from_optimal_solution(
        self,
        feature_matrix: np.ndarray,
        single_trip_map: dict,
        trip_list: list,
        trip_costs: list[TripCost],
        vehicle_to_trips_cost_map: dict,
        trip_to_vehicle_cost_map: dict,
        request_batch: list,
        active_requests: dict,
        reject_vehicle_ids: list[int] | None = None,
    ) -> None:
        """
        Compute and store the Fenchel-Young loss for the current iteration.

        Uses the CO_TripCostMinimization solution as y_star (ground truth) and
        CO_ScoreMaximization as the MAP oracle for perturb-and-MAP.

        The result is stored in self.last_loss.  If the ILP did not yield a
        feasible solution (y_star is all-zero), the loss is skipped and
        self.last_loss is set to None.

        Side effects:
            Sets self.last_loss.
        """
        reject_vehicle_ids = reject_vehicle_ids or []
        imitation_scores, trip_vehicle_ids = self._build_imitation_scores_with_reject(
            trip_costs=trip_costs,
            request_batch=request_batch,
            reject_vehicle_ids=reject_vehicle_ids,
            vehicle_to_trips_cost_map=vehicle_to_trips_cost_map,
        )
        y_star = self._build_y_star_from_imitation_scores(
            imitation_scores=imitation_scores,
            trip_vehicle_ids=trip_vehicle_ids,
            reject_vehicle_ids=reject_vehicle_ids,
        )

        if y_star.sum().item() == 0:
            console_logger.warning("FY loss skipped: Optimizer did not assign any RTVs.")
            self.last_loss = None
            raise RuntimeError("Y_star must have some values as we consider the reject action separately. The score optimizer must make a decision of some sort.")

        feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        scores = self.model(feature_tensor)    

        oracle = make_map_oracle(
            self.coaml_optimizer,
            single_trip_map,
            trip_list,
            trip_costs,
            vehicle_to_trips_cost_map,
            trip_to_vehicle_cost_map,
            request_batch,
            active_requests,
            self.config,
            reject_vehicle_ids=reject_vehicle_ids,
        )

        self.last_loss = self.fy_loss(scores, y_star, oracle)
        console_logger.info(f"COAML FY loss computed: {self.last_loss.item():.4f}")

    def _log_assignment_status(self, result: AssignmentResult, unserved_requests: set[int], current_time: float):
        assignment_status = {PayloadKeys.STATS_ASSIGNED: result.request_assignment, 
                            PayloadKeys.STATS_UNSERVED: list(unserved_requests)}
        data_logger.info("Status", extra={"timestamp": current_time, "status": assignment_status})  