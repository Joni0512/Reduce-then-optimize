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
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model, CandidateConflictGraphBuilder
# keep these constants for compatibility with commented out code in coaml_pipeline.py
from rtv_solver.pipeline.imitation_handler import (
    ImitationHandler,
    TYPE_BEST_ORDERED_MATCH,
    TYPE_BEST_UNORDERED_MATCH,
    SCORING_RULE_LEGACY,
)

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
            model: ScoringMLP | torch.nn.Module | None = None,
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
        # 2026-07-30: additive - config.MODEL_TYPE selects "mlp" (ScoringMLP,
        # default, unchanged) or "gnn" (CandidateScoringGNN) when no explicit
        # `model=` is passed in. See build_scoring_model and self._score below.
        self.model = model if model is not None else build_scoring_model(
            config.MODEL_TYPE,
            feature_dim=FeatureBuilder.FEATURE_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            num_message_passing_layers=config.GNN_NUM_MESSAGE_PASSING_LAYERS,
            aggregator=config.GNN_AGGREGATOR,
        )
        self._candidate_graph_builder = CandidateConflictGraphBuilder()
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
        trip_costs: list[TripCost],
        requests: list,
        active_requests: dict,
        reject_vehicle_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Build a globally feasible binary target vector y_star by maximizing the
        imitation scores using the existing assignment ILP.

        The imitation scores describe how well each generated vehicle-trip candidate
        matches the expert solution. However, selecting the highest-scoring candidate
        for each vehicle independently is not sufficient, because candidates belonging
        to different vehicles may contain the same request. This can produce an invalid
        target in which one request is assigned to multiple vehicles.

        Instead, this method reuses the existing assignment ILP that is also used
        during inference. Rather than optimizing the neural-network prediction scores,
        the ILP objective is replaced by the imitation scores. This produces the
        globally best feasible assignment with respect to the expert imitation scores.

        The ILP enforces all assignment constraints, including:

        - each vehicle selects at most one trip or reject action;
        - each request is assigned at most once across all vehicles;
        - only feasible vehicle-trip combinations can be selected;
        - active-request constraints are respected when KEEP_ACTIVE is enabled.

        The resulting solver decisions are converted into a binary vector with the same
        layout as the input score vector:

            [trip decisions..., reject-action decisions...]

        where 1 indicates that an action is part of the globally optimal feasible
        assignment and 0 indicates that it is not selected.

        2026-07-27: under SCORING_RULE_LEGACY, positively-scored trips are exact
        full-subsequence matches against each vehicle's OWN expert continuation, so
        two vehicles can never have a positive score on the same request (the
        expert manifest assigns each request to exactly one vehicle) - the cross-
        vehicle collision this ILP exists to prevent is structurally impossible for
        legacy. So for legacy, skip the (comparatively expensive) global ILP and use
        the cheap independent per-vehicle selection instead
        (ImitationHandler.build_y_star_per_vehicle_from_imit_scores). exponential_prefix
        keeps the global ILP: its partial-credit scoring can score a trip positively
        for a vehicle even when the trip carries a request truly owned by a
        different vehicle (only the matching prefix is checked), so cross-vehicle
        collisions are a real risk there.
        """
        reject_vehicle_ids = reject_vehicle_ids or []
        if self.config.IMITATION_SCORING_RULE == SCORING_RULE_LEGACY:
            trip_vehicle_ids = [trip_cost.vehicle_id for trip_cost in trip_costs]
            return ImitationHandler.build_y_star_per_vehicle_from_imit_scores(
                imitation_scores_with_reject=imitation_scores,
                trip_vehicle_ids=trip_vehicle_ids,
                reject_vehicle_ids=reject_vehicle_ids,
            )

        trip_count = len(trip_costs)

        expected_length = trip_count + len(reject_vehicle_ids)
        if imitation_scores.shape[0] != expected_length:
            raise ValueError(
                f"Imitation-score length mismatch: got "
                f"{imitation_scores.shape[0]}, expected {expected_length}."
            )

        #The score vector is ordered as: [scores for generated vehicle-trip candidates,
        #scores for vehicle-specific reject actions].
        # The ILP expects trip and reject scores separately.
        trip_scores = (
            imitation_scores[:trip_count]
            .detach()
            .cpu()
            .numpy()
        )

        reject_scores = (
            imitation_scores[trip_count:]
            .detach()
            .cpu()
            .numpy()
        )

        # Solve one global assignment problem over all candidate trips and reject actions.
        # The ILP maximizes the total imitation score while enforcing feasibility:
        # each vehicle can select at most one action and each request can be assigned
        # to at most one vehicle.

        ilp_model, x_t, x_r, x_reject = self.coaml_optimizer.solve_ilp(
            trip_scores,
            requests,
            active_requests,
            penalty=self.config.ILP_PENALTY,
            keep_active=self.config.KEEP_ACTIVE,
            reject_action_scores=reject_scores,
            reject_vehicle_ids=reject_vehicle_ids,
        )

        # First part of y_star: selected normal trips.
        # x_t[idx] is a binary ILP variable. The 0.5 threshold robustly converts
        # the solver's floating-point output into a binary target (0 or 1).
        y_trip = torch.tensor(
            [
                1 if x_t[idx].X > 0.5 else 0
                for idx in range(trip_count)
            ],
            dtype=imitation_scores.dtype,
            device=imitation_scores.device,
        )

        # Second part: selected vehicle-specific reject actions.
        y_reject = torch.tensor(
            [
                1 if (
                    vehicle_id in x_reject
                    and x_reject[vehicle_id].X > 0.5
                ) else 0
                for vehicle_id in reject_vehicle_ids
            ],
            dtype=imitation_scores.dtype,
            device=imitation_scores.device,
        )

        y_star = torch.cat([y_trip, y_reject], dim=0)

        if y_star.shape != imitation_scores.shape:
            raise ValueError(
                f"y_star shape mismatch: got {y_star.shape}, "
                f"expected {imitation_scores.shape}."
            )

        return y_star
    # old version of this function is commented out below but is still used for reference
    #def _build_y_star_from_imitation_scores(
    #    self,
    #    imitation_scores: torch.Tensor,
    #    trip_vehicle_ids: list[int],
    #    reject_vehicle_ids: list[int] | None = None,
    #) -> torch.Tensor:
        #"""
       # Convert imitation scores into binary y* according to configured strategy.
       # """
       # reject_vehicle_ids = reject_vehicle_ids or []
       #if self.config.Y_STAR_TYPE == TYPE_BEST_ORDERED_MATCH:
        #    y_star = ImitationHandler.build_y_star_per_vehicle_from_imit_scores(
        #        imitation_scores_with_reject=imitation_scores,
        #        trip_vehicle_ids=trip_vehicle_ids,
        #        reject_vehicle_ids=reject_vehicle_ids,
        #    )
        #    if y_star.shape[0] != imitation_scores.shape[0]:
        #        raise ValueError(f"y_star shape mismatch: got {y_star.shape[0]}, expected {imitation_scores.shape[0]}.")
        #    if not torch.all((y_star == 0) | (y_star == 1)):
        #        raise ValueError("y_star must be binary (0/1).")
        #    return y_star
       # raise ValueError(f"Invalid y_star type: {self.config.Y_STAR_TYPE}")

    def _measure_pruner_pair_recall(
        self,
        trip_handler: TripHandler,
        request_batch: list,
    ) -> dict | None:
        """
        2026-07-12: diagnostic added to check whether RequestGraphPruner (run inside
        trip_handler.run(), see trip_handler.allowed_request_pairs) discards request
        pairs that belong to the offline-optimal solution loaded by ImitationHandler.
        Motivation: since ImitationHandler only ever scores the trips that survive
        pruning, a pruned-away optimal pair silently degrades the y* target for that
        vehicle (see ImitationHandler.count_fallback_vehicles), this measures the
        upstream cause instead of only the downstream symptom.

        Ground-truth pairs are "requests served by the same vehicle in the optimal
        solution" -- the same definition RequestGraphLabelBuilder.
        build_positive_pairs_from_solution_payload uses to build the pruner's own
        training labels, so recall here is directly comparable to the pruner's
        offline eval metrics (see evaluate_pruner_metrics_all.py). Built from
        self.imitation_handler.optimal_solution directly (already-parsed
        {vehicle_id: [request_ids]}) rather than re-parsing the raw payload, to avoid
        a second, possibly inconsistent, payload-key parsing path.

        Returns None when pruning is disabled (trip_handler.allowed_request_pairs is
        None) or when this batch has no optimal pairs to compare against, since
        recall is undefined in both cases.
        """
        allowed_pairs = trip_handler.allowed_request_pairs
        if allowed_pairs is None:
            return None

        batch_ids = {request.id for request in request_batch}
        optimal_pairs: set[tuple[int, int]] = set()
        for ordered_request_ids in self.imitation_handler.optimal_solution.values():
            vehicle_batch_ids = [rid for rid in ordered_request_ids if rid in batch_ids]
            for i in range(len(vehicle_batch_ids)):
                for j in range(i + 1, len(vehicle_batch_ids)):
                    optimal_pairs.add(tuple(sorted((vehicle_batch_ids[i], vehicle_batch_ids[j]))))

        if not optimal_pairs:
            return None

        preserved_pairs = optimal_pairs & allowed_pairs
        return {
            PayloadKeys.STATS_PRUNER_OPTIMAL_PAIRS_TOTAL: len(optimal_pairs),
            PayloadKeys.STATS_PRUNER_OPTIMAL_PAIRS_PRESERVED: len(preserved_pairs),
            PayloadKeys.STATS_PRUNER_PAIR_RECALL: len(preserved_pairs) / len(optimal_pairs),
        }

    def _log_pruner_imitation_diagnostics(
        self,
        pruner_recall_stats: dict | None,
        fallback_vehicle_count: int,
        vehicles_considered: int,
        current_time: float,
    ) -> None:
        """
        2026-07-12: combines _measure_pruner_pair_recall (upstream: did pruning drop an
        optimal pair) with ImitationHandler.count_fallback_vehicles (downstream: did a
        vehicle's y* degrade to the no-positive-candidate fallback) into one log record,
        so both can be correlated per iteration without re-deriving either from raw logs.
        Purely observational -- no effect on training or assignment.
        """
        diagnostics = {
            PayloadKeys.STATS_PRUNER_FALLBACK_VEHICLES: fallback_vehicle_count,
            PayloadKeys.STATS_PRUNER_VEHICLES_CONSIDERED: vehicles_considered,
        }
        if pruner_recall_stats is not None:
            diagnostics.update(pruner_recall_stats)

        data_logger.info("PrunerImitationDiagnostics", extra={"timestamp": current_time, "stats": diagnostics})

        if pruner_recall_stats is not None:
            console_logger.info(
                f"Pruner/imitation diagnostics: pair recall "
                f"{pruner_recall_stats[PayloadKeys.STATS_PRUNER_PAIR_RECALL]:.3f} "
                f"({pruner_recall_stats[PayloadKeys.STATS_PRUNER_OPTIMAL_PAIRS_PRESERVED]}/"
                f"{pruner_recall_stats[PayloadKeys.STATS_PRUNER_OPTIMAL_PAIRS_TOTAL]} optimal pairs kept), "
                f"fallback vehicles {fallback_vehicle_count}/{vehicles_considered}"
            )
        elif fallback_vehicle_count > 0:
            console_logger.info(
                f"Pruner/imitation diagnostics: pruner disabled or no optimal pairs in batch, "
                f"fallback vehicles {fallback_vehicle_count}/{vehicles_considered}"
            )

    def _score(
        self,
        feature_tensor: torch.Tensor,
        trip_costs: list[TripCost],
        reject_vehicle_ids: list[int],
    ) -> torch.Tensor:
        """
        Score `feature_tensor` (rows: trip candidates..., then reject actions)
        with self.model, building a CandidateGraph first if the model needs one.

        2026-07-30: single choke point for both self.model(...) call sites in
        this file (the main ILP-scoring call below and the FY-loss call in
        _compute_fy_loss_from_optimal_solution), so ScoringMLP and
        CandidateScoringGNN can be swapped via config.MODEL_TYPE without
        duplicating the branch. FenchelYoungLoss/make_map_oracle only ever
        perturb and re-solve over the already-computed score vector - they
        never call self.model again - so this one call site per iteration is
        sufficient; no edge_index needs to be threaded any deeper.
        """
        if not getattr(self.model, "requires_graph", False):
            return self.model(feature_tensor)

        graph = self._candidate_graph_builder.build(
            trip_costs, reject_vehicle_ids, device=feature_tensor.device
        )
        return self.model(feature_tensor, graph.edge_index)

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
            # 2026-07-16: current_time now passed through - USE_REQUEST_PRUNER needs it
            # for its urgency force-keep rule (see TripHandler.run()'s ValueError guard).
            # payload_object.current_time already exists at this point (used further down
            # for feature_builder.build_matrix_from_trip_handler), just wasn't threaded here yet.
            trip_handler = TripHandler(
                vehicle_handler.vehicles,
                request_batch,
                active_requests,
                iteration,
                self.config,
                current_time=payload_object.current_time)
        except Exception as e:
            raise e
        
        loss_tracked = False

        if len(vehicle_handler.vehicles) != 0:
            single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map = trip_handler.run()

            # 2026-07-12: measure, right after pruning and before any ML scoring happens,
            # whether the request-graph pruner (TripHandler.run() -> RequestGraphPruner)
            # already dropped request pairs that belong to the offline-optimal solution
            # for this batch. Diagnostic-only: does not affect trip_costs or the
            # assignment. See _measure_pruner_pair_recall for the exact definition.
            pruner_recall_stats = self._measure_pruner_pair_recall(trip_handler, request_batch)

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
                feature_scores_with_reject = self._score(
                    feature_tensor_with_reject, trip_costs, reject_vehicle_ids
                )
                num_reject_actions = len(reject_vehicle_ids)
                if num_reject_actions > 0:
                    # Respect the same ordering as in add_reject_action_entries():
                    # [trip rows..., reject rows...].
                    feature_scores = feature_scores_with_reject[:-num_reject_actions]
                    reject_action_scores = feature_scores_with_reject[-num_reject_actions:]
                else:
                    feature_scores = feature_scores_with_reject
                    reject_action_scores = torch.empty(0, dtype=feature_scores_with_reject.dtype)
                
                # 2026-07-16: use trip_handler.requests (post-RequestPruner list), NOT the
                # raw request_batch, for both calls below. solve_ilp() does
                # self.single_trip_map[request.id] for every request it's given (see
                # CO_ScoreMaximization.solve_ilp) - single_trip_map only has entries for
                # requests that survived TripHandler.run()'s pruning step, so passing the
                # full, unpruned request_batch here raises a KeyError the moment
                # USE_REQUEST_PRUNER actually removes something (found by running the
                # request pruner through mode="optimal" for the first time - see
                # run_opt_single_instance.py). transform_solution_to_assignment() must be
                # given the SAME list in the SAME order as solve_ilp(), since x_r's indices
                # are positional against that requests list.
                ilp_model, x_t, x_r, x_reject = self.coaml_optimizer.solve_ilp(
                    feature_scores,
                    trip_handler.requests,
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
                    trip_handler.requests,
                    x_reject=x_reject,
                    reject_vehicle_ids=reject_vehicle_ids,
                )
                # 2026-07-16: request_batch (full, NOT trip_handler.requests) is correct
                # here on purpose - this only reads which requests the OFFICIAL optimal
                # solution assigns per vehicle (_measure_pruner_pair_recall /
                # score_trip_costs_against_optimal_by_vehicle), it never indexes into
                # single_trip_map, so there's no KeyError risk. Using the full batch means
                # a request pruned away before trip generation still correctly shows up as
                # "not matched by any candidate trip" in the imitation score, instead of
                # silently vanishing from the ground truth.
                imitation_scores_with_reject, trip_vehicle_ids = self._build_imitation_scores_with_reject(
                    trip_costs=trip_costs,
                    request_batch=request_batch,
                    reject_vehicle_ids=reject_vehicle_ids,
                    vehicle_to_trips_cost_map=vehicle_to_trips_cost_map,
                )

                # 2026-07-12: downstream half of the pruner/imitation diagnostic. Counts
                # vehicles whose imitation scores are all <= 0 (build_y_star_per_vehicle_
                # from_imit_scores would fall back to picking the minimum instead of a
                # genuine positive match). Combined with pruner_recall_stats above, this
                # lets us tell whether a low-recall pruning outcome is actually causing
                # degraded imitation targets, instead of just guessing. Read-only, does
                # not change y_star.
                fallback_vehicle_count, vehicles_considered = ImitationHandler.count_fallback_vehicles(
                    imitation_scores_with_reject=imitation_scores_with_reject,
                    trip_vehicle_ids=trip_vehicle_ids,
                    reject_vehicle_ids=reject_vehicle_ids,
                )
                self._log_pruner_imitation_diagnostics(
                    pruner_recall_stats,
                    fallback_vehicle_count,
                    vehicles_considered,
                    payload_object.current_time,
                )

                # Construct a globally feasible imitation target by maximizing the imitation
                # scores with the assignment ILP instead of selecting the best trip per vehicle.
                y_star = self._build_y_star_from_imitation_scores(
                    imitation_scores=imitation_scores_with_reject,
                    trip_costs=trip_costs,
                    requests=trip_handler.requests,
                    active_requests=active_requests,
                    reject_vehicle_ids=reject_vehicle_ids,
                )
                #y_star = self._build_y_star_from_imitation_scores(
                #    imitation_scores=imitation_scores_with_reject,
                #    trip_vehicle_ids=trip_vehicle_ids,
                #    reject_vehicle_ids=reject_vehicle_ids,
                #)

                # calculate the true optimal solution based on the imitation scores
                # 2026-07-16: request_batch (full) here too - transform_optimal_solution_
                # to_assignment() only uses `requests` for a length/logging count
                # (request_count - trip_count = unassigned_trip_count), it never indexes
                # single_trip_map, so no KeyError risk. Using the full batch means a
                # pruned-away request correctly counts as "unassigned this round" here too,
                # consistent with how `unserved_requests` is computed further down from the
                # full request_batch as well.
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
            # 2026-07-16: same trip_handler.requests fix as coaml_optimizer above -
            # CO_TripCostMinimization.solve_ilp() also does self.single_trip_map[request.id]
            # per request, so this crashes with the raw request_batch too once pruning
            # actually removes a request.
            trip_obj_scores = np.fromiter((tc.cost for tc in trip_costs), dtype=float, count=len(trip_costs))
            default_ilp_model, default_x_t, default_x_r = self.default_optimizer.solve_ilp(
                trip_obj_scores,
                trip_handler.requests,
                active_requests,
                penalty=self.config.ILP_PENALTY,
                keep_active=self.config.KEEP_ACTIVE)
            default_result = self.default_optimizer.transform_solution_to_assignment(
                default_ilp_model,
                default_x_t,
                default_x_r,
                trip_handler.requests,
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
            if len(feature_tensor_with_reject) > 0:
                if mode == "train":
                    result = optimal_result
                elif mode == "eval":
                    result = score_result
                elif mode == "optimal": 
                    # allows us to run the validation files but record the optimal steps and see how close to the optimal it itself can even get
                    result = optimal_result
                elif mode == "offline":
                    # allows us to run the offline pipeline decision while training but never used
                    result = default_result
                else:
                    raise ValueError(f"Invalid mode: {mode}")
            else:
                # No NN/scoring rows: do not apply default ILP as stand-in (keeps eval about the scorer only).
                result = AssignmentResult.get_empty_result(
                    unassigned_trip_count=len(request_batch),
                )

            # compute Fenchel-Young loss from known optimal solution
            if len(feature_tensor_with_reject) > 0:
                # 2026-07-16: pass trip_handler.requests separately from request_batch -
                # see _compute_fy_loss_from_optimal_solution's docstring for why it needs
                # both (imitation-score ground truth vs. the MAP oracle's ILP calls have
                # different, incompatible requirements for which list to use).
                self._compute_fy_loss_from_optimal_solution(
                    feature_matrix_with_reject,
                    single_trip_map,
                    trip_list,
                    trip_costs,
                    vehicle_to_trips_cost_map,
                    trip_to_vehicle_cost_map,
                    request_batch,
                    trip_handler.requests,
                    active_requests,
                    reject_vehicle_ids,
                )
                loss_tracked = True

            if self.config.REBALANCING:
                rebalancing_optimizer = CO_RebalancingCoverage(self.config)
                result = rebalancing_optimizer.run(result, vehicle_handler.vehicles, request_batch)

        else:
            result = AssignmentResult.get_empty_result(
                unassigned_trip_count=len(request_batch),
            )

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
        pruned_requests: list,
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

        2026-07-16: takes TWO different request lists on purpose, because the two
        things this method does need different ones:
        - `request_batch` (full, pre-pruning): used only to look up which requests
          the OFFICIAL optimal solution assigns per vehicle
          (_build_imitation_scores_with_reject -> score_trip_costs_against_optimal_
          by_vehicle). Never indexed into single_trip_map, so the full batch is
          safe AND correct here - a request pruned away before trip generation
          should still show up as "not matched by any candidate trip".
        - `pruned_requests` (== trip_handler.requests, post-RequestPruner): used
          for the MAP oracle (make_map_oracle -> CO_ScoreMaximization.solve_ilp),
          which does self.single_trip_map[request.id] per request. Passing the
          full request_batch here raises a KeyError the moment the request pruner
          actually removes something, exactly like the coaml_optimizer/
          default_optimizer calls in solve_iteration() above - same underlying
          cause, just a third call site that needed the same fix.

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

        # Build the globally optimal feasible imitation target using the assignment ILP.
        y_star = self._build_y_star_from_imitation_scores(
            imitation_scores=imitation_scores,
            trip_costs=trip_costs,
            requests=pruned_requests,
            active_requests=active_requests,
            reject_vehicle_ids=reject_vehicle_ids,
        )
        # old method of building y_start from imitation scores, now replaced by the above MLP
        #y_star = self._build_y_star_from_imitation_scores(
        #    imitation_scores=imitation_scores,
        #    trip_vehicle_ids=trip_vehicle_ids,
        #    reject_vehicle_ids=reject_vehicle_ids,
        #)

        if y_star.sum().item() == 0:
            console_logger.warning("FY loss skipped: Optimizer did not assign any RTVs.")
            self.last_loss = None
            raise RuntimeError("Y_star must have some values as we consider the reject action separately. The score optimizer must make a decision of some sort.")

        feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        scores = self._score(feature_tensor, trip_costs, reject_vehicle_ids)

        # pruned_requests, not request_batch - see this method's docstring for why.
        oracle = make_map_oracle(
            self.coaml_optimizer,
            single_trip_map,
            trip_list,
            trip_costs,
            vehicle_to_trips_cost_map,
            trip_to_vehicle_cost_map,
            pruned_requests,
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