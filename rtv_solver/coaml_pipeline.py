import multiprocessing
import sys
from multiprocessing import Pool
import numpy as np
import copy
from typing import Any, Optional
import torch

from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.online_rtv_solver import OnlineRTVSolver

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
            offline_payload: dict
        ):
        """
        Initialize the COAML pipeline solver.

        Parameters:
            - config: Config object
            - offline_payload: Offline payload object in order to calculate feature normalization values and get structure of feature matrix
        """
        # TODO add elements to config in order to make it more readable
        self.config = config
        self.offline_payload = offline_payload
        self.feature_builder = FeatureBuilder(offline_payload, self.config)
        self.model = ScoringMLP(feature_dim=FeatureBuilder.FEATURE_SIZE, hidden_dim=32)
        self.fy_loss = FenchelYoungLoss(num_samples=15, sigma=0.1)
        self.imitation_handler = ImitationHandler(config)
        
        self.coaml_optimizer = CO_ScoreMaximization(config)
        self.default_optimizer = CO_TripCostMinimization(config)    
        
        self.last_loss: Optional[torch.Tensor] = None
        self.loss_history: list[Optional[float]] = []
        
        if sys.platform == "darwin": # required to run correctly on MacOS
            try:
                # it is required here as we do not call OnlineRTVSolver as an object
                multiprocessing.set_start_method("fork")
            except RuntimeError: # start method was already set somewhere else -> don't crash
                pass

    # def manage_training(self):
    #     """
    #     Manage the training of the pipeline.
    #     """
    #     # TODO implement this method
    #     for epoch in config.epochs:
    #         for episode in episodes:
    #             self.train(episode)

    #         if episode % config.VALIDATION_INTERVAL == 0:    
    #         self.validate(episode)
    #     pass

    def train(self):
        """
        Single training run for the payload.  
        """
        # TODO implement this method
        pass

    def validate(self):
        """
        Single validation run for the payload.
        """
        # TODO implement this method
        pass

    def solve_pdptw(self, payload: dict):
        """
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

        driver_runs = payload[PayloadParser.DRIVERS]

        while current_time < end_time:
            console_logger.info(f"=== Iteration {iteration} offline RTV Solver at time {current_time} ===")
            
            # select requests that are to be considered in the current interval with pickup_window [current_time, current_time + interval]
            selected_requests = {}
            for request in payload[PayloadParser.REQUESTS]:
                # if start of time window is part of current batch_interval
                if (request[PayloadParser.REQ_PICKUP_WINDOW_START] >= current_time and 
                    request[PayloadParser.REQ_PICKUP_WINDOW_START] < current_time + self.config.BATCH_INTERVAL 
                    ):
                    selected_requests[request[PayloadParser.REQ_BOOKING_ID]] = request
            
            # remove requests that are already part of vehicles; covered by PayloadParser in OnlineRTVsolver # TODO check
            for dr in driver_runs:
                manifest = dr[PayloadParser.DRIVER_MANIFEST]
                for stop in manifest:
                    if stop[PayloadParser.MANIFEST_BOOKING_ID] in selected_requests:
                        del selected_requests[stop[PayloadParser.MANIFEST_BOOKING_ID]]
            selected_requests = list(selected_requests.values())

            # create a new payload with the selected requests
            new_payload = {
                PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
                PayloadParser.REQUESTS: selected_requests,
                PayloadParser.DRIVERS: driver_runs,
                PayloadParser.CURRENT_TIME: current_time}

            # solve the RTV problem and update manifests
            if len(selected_requests) == 0:
                new_driver_runs = driver_runs
            else:    
                new_driver_runs = self.solve_iteration(new_payload, iteration)
                          
            # increment time (might not be the size of the batch) and iteration
            current_time += self.config.STEP_SIZE 
            iteration += 1

            # update vehicles based on decisions in the previous step until current time (might not be the entire interval)
            simulated_driver_runs = OnlineRTVSolver.simulate_manifest(self.config, current_time, new_driver_runs,   intermediate_location=True) # TODO check if intermediate_location is correct
            driver_runs = simulated_driver_runs

        final_driver_runs = OnlineRTVSolver.finalize_driverRuns(self.config, driver_runs, payload[PayloadParser.DEPOT])
        return final_driver_runs
    
    def solve_iteration(self, subset_payload, iteration = 0):
        """
        Solver for the entire payload that is given, based on the onlineRTVSolver but adapted to COAML pipeline.
        """
        # TODO (major effort) improve code quality as we currently have a lot of repetition that should not be required as we need similar information in online, offline and COAML
        
        # initalize network and payload
        NetworkHandler.init(True, self.config.SERVER_URL)
        payload_object = PayloadParser.get_payload_object(subset_payload, False)
        request_handler = RequestHandler(payload_object.requests, 
                                         self.config.DWELL_PICKUP, 
                                         self.config.DWELL_ALIGHT)
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
                                                 boarded_trips, 
                                                 self.config.DWELL_ALIGHT, 
                                                 self.config.DWELL_PICKUP)
        
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
            feature_matrix, _ = self.feature_builder.build_matrix_from_trip_handler(trip_handler, payload_object.current_time) 
            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
            if len(feature_tensor) > 0:
                feature_scores = self.model(feature_tensor)
                ilp_model, x_t, x_r = self.coaml_optimizer.solve_ilp(
                    feature_scores,
                    request_batch, 
                    active_requests,
                    penalty=self.config.ILP_PENALTY,
                    keep_active=self.config.KEEP_ACTIVE)

                # NOTE this is for now just experimental
                
                trip_combinations = self.imitation_handler.tripCosts_to_request_combinations(trip_costs)
                optimal_solution = self.imitation_handler.get_optimal_request_solution_for_batch(request_batch)
                imitation_scores = self.imitation_handler.score_combinations_against_solution(trip_combinations, optimal_solution) 

                print(f"Complete solution: {self.imitation_handler.optimal_solution}")
                for idx, (ml_score, tc, imitation_score) in enumerate[TripCost](zip(feature_scores,trip_costs, imitation_scores)):
                    print(f"x_t: {x_t[idx].X}, score: {ml_score}, imit:{imitation_score.item()}, tc: {tc.simple_str()}, ordered_stop_sequence: {tc.get_ordered_stop_sequence()}")

                # transform solution to assignment, used to update vehicles and manifests
                # THIS result must be used for future comparisons
            
            # calculate default solution based on the ILP minimizing trip costs (used for FY loss)
            trip_obj_scores = np.fromiter((tc.cost for tc in trip_costs), dtype=float, count=len(trip_costs))
            default_ilp_model, default_x_t, default_x_r = self.default_optimizer.solve_ilp(
                trip_obj_scores, 
                request_batch, 
                active_requests, 
                penalty=self.config.ILP_PENALTY, 
                keep_active=self.config.KEEP_ACTIVE)
            console_logger.info(f"Default ILP solved in {default_ilp_model.Runtime:.3f} s.")
            default_result = self.default_optimizer.transform_solution_to_assignment(
                default_ilp_model, default_x_t, default_x_r, request_batch)

            # train mode, keep on right track - decide which run should move forward
            result = default_result

            # compute Fenchel-Young loss from known optimal solution
            if True and len(feature_tensor) > 0:
                self._compute_fy_loss_from_optimal_solution(
                    feature_matrix,
                    single_trip_map, 
                    trip_list, 
                    trip_costs,
                    vehicle_to_trips_cost_map, 
                    trip_to_vehicle_cost_map,
                    request_batch, 
                    active_requests)
                loss_tracked = True
            else:
                # compute Fenchel-Young loss from default ILP solution
                self._compute_fy_loss_from_default_ilp(
                    feature_matrix,
                    default_ilp_model,
                    default_x_t,
                    single_trip_map,
                    trip_list,
                    trip_costs,
                    vehicle_to_trips_cost_map,
                    trip_to_vehicle_cost_map,
                    request_batch,
                    active_requests)
                loss_tracked = True

            if self.config.REBALANCING:
                rebalancing_optimizer = CO_RebalancingCoverage(self.config)
                result = rebalancing_optimizer.run(result, vehicle_handler.vehicles, request_batch)

        if not loss_tracked:
            self.last_loss = None

        if self.last_loss is not None:
            current_loss = float(self.last_loss.detach().cpu().item())
            self.loss_history.append(current_loss)
            console_logger.info(f"Tracked model loss at time {payload_object.current_time}: {current_loss:.4f}")
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
            subset_payload[PayloadParser.DRIVERS], 
            updated_driver_runs,
            unserved_requests, 
            subset_payload[PayloadParser.REQUESTS],
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

        Uses the CO_TripCostMinimization solution as y_star (ground truth) and
        CO_ScoreMaximization as the MAP oracle for perturb-and-MAP.

        The result is stored in self.last_loss.  If the ILP did not yield a
        feasible solution (y_star is all-zero), the loss is skipped and
        self.last_loss is set to None.

        Side effects:
            Sets self.last_loss.

        # TODO update the y_start calculation from the imitationHandler
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

        # TODO update the y_start calculation from the imitationHandler
        """
        # TODO this currently handles only a single vehicle
        trip_combinations = self.imitation_handler.tripCosts_to_request_combinations(trip_costs)
        optimal_solution_from_batch = self.imitation_handler.get_optimal_request_solution_for_batch(request_batch)
        imitation_scores = self.imitation_handler.score_combinations_against_solution(trip_combinations, optimal_solution_from_batch) 
        
        if self.config.Y_STAR_TYPE == TYPE_BEST_ORDERED_MATCH:
            y_star = ImitationHandler.get_y_star_best_ordered_match(imitation_scores)
        elif self.config.Y_STAR_TYPE == ImitationHandler.TYPE_BEST_UNORDERED_MATCH:
            y_star = ImitationHandler.get_y_star_best_unordered_match(imitation_scores)
        else:
            raise ValueError(f"Invalid y_star type: {self.config.Y_STAR_TYPE}")

        for idx, (tc, imitation_score, y_star_value) in enumerate(zip (trip_costs, imitation_scores, y_star)):
                print(f"y_star: {y_star_value},  score: {imitation_score}, tc: {tc.simple_str()}, order: {tc.get_ordered_stop_sequence()}")

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
        console_logger.info(f"COAML FY loss computed: {self.last_loss.item():.4f}")

    def _log_assignment_status(self, result: AssignmentResult, unserved_requests: set[int], current_time: float):
        assignment_status = {PayloadParser.STATS_ASSIGNED: result.request_assignment, 
                            PayloadParser.STATS_UNSERVED: list(unserved_requests)}
        data_logger.info("Status", extra={"timestamp": current_time, "status": assignment_status})  