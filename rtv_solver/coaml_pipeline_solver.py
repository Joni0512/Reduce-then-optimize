import multiprocessing
import sys
from multiprocessing import Pool
import numpy as np
import copy
from typing import Optional
import torch

from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.online_rtv_solver import OnlineRTVSolver
from rtv_solver.structure.config import Config

from rtv_solver.pipeline import CO_ScoreMaximization, CO_TripCostMinimization, CO_RebalancingCoverage, FeatureBuilder
from rtv_solver.pipeline import FenchelYoungLoss, make_map_oracle, extract_y_binary, ScoringMLP

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
        config: Config = None,
    ):
        self.config = config
        self.model = ScoringMLP(feature_dim=66, hidden_dim=32) # TODO rebuild so we can have the fixed feature_matrix dimension here and define them before instead of inline to
        self.fy_loss = FenchelYoungLoss(num_samples=2, sigma=1.0)
        # Holds the FY loss tensor from the most recent solve_iteration call.
        # None when no model is set or when the ILP returned no feasible solution.
        self.last_loss: Optional[torch.Tensor] = None
        # it is required here as we do not call OnlineRTVSolver as an object
        if sys.platform == "darwin": # required to run online_solver correctly on MacOS
            try:
                multiprocessing.set_start_method("fork")
            except RuntimeError: # start method was already set somewhere else -> don't crash
                pass

    def solve_pdptw(self, payload: dict):
        """
        Handles payloads across iterations of the rolling horizon
        
        With config.return_depot, this method will not add the final trips to the depot. The user has to call finalize_driverRuns(...) to add the final stops.
        """
        self.feature_builder = FeatureBuilder(payload, self.config)
        # determine time interval of entire requests set
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
        request_handler = RequestHandler(payload_object.requests, self.config.DWELL_PICKUP, self.config.DWELL_ALIGHT)
        
        # filter active and boarded requests for subsequent action as they need to be integrated when handling new trip generation
        temp_batch = request_handler.get_all_requests()
        request_batch = []
        active_requests = {}
        boarded_requests = {}
        for req in temp_batch:
            req_id = req.id
            if req_id in payload_object.boarded_requests_keys:
                boarded_requests[req_id] = req
            else:
                if req_id in payload_object.active_requests_keys:
                    active_requests[req_id] = req
                request_batch.append(req)
        
        # create trips of all already boarded requests # NOTE these requests are not always boarded at this point but might be still committed to a vehicle (especially if it is the first request of an idling vehicle)
        boarded_trips = TripHandler.create_trip_for_picked_requests(boarded_requests, iteration)
        
        # initialize all vehicles as they are stored in the payload-object
        vehicle_handler = VehicleHandler(payload_object.depot, 
                                         payload_object.driver_runs,
                                         self.config)
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
        
        if len(vehicle_handler.vehicles) != 0:
            single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map = trip_handler.run()

            feature_matrix, feature_names = self.feature_builder.build_matrix_from_trip_handler(trip_handler, payload_object.current_time)  
            
            # TODO make optimizer injectable to allow for different optimizers (e.g. CO_TripCostMinimization, CO_RebalancingCoverage, etc.) and handle setup in the main file instead of here, so we can run it based on the mode of the current program
            self.optimizer = CO_ScoreMaximization(
                single_trip_map, 
                trip_list, 
                trip_costs, 
                vehicle_to_trips_cost_map, 
                trip_to_vehicle_cost_map, 
                self.config)
            # self.optimizer = CO_TripCostMinimization(
            #     single_trip_map, 
            #     trip_list, 
            #     trip_costs, 
            #     vehicle_to_trips_cost_map, 
            #     trip_to_vehicle_cost_map, 
            #     self.config)

            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
            feature_scores = self.model(feature_tensor)
            
            # Split run() so we can capture x_t for y_star extraction; result is identical to self.optimizer.run().
            ilp_model, x_t, x_r = self.optimizer.solve_ilp(
                feature_scores,
                request_batch, active_requests,
                penalty=self.config.ILP_PENALTY,
                keep_active=self.config.KEEP_ACTIVE,
            )
            result = self.optimizer.transform_solution_to_assignment(
                ilp_model, x_t, x_r, request_batch
            )
            self._compute_fy_loss(
                feature_matrix,
                ilp_model, x_t, len(trip_costs),
                single_trip_map, trip_list, trip_costs,
                vehicle_to_trips_cost_map, trip_to_vehicle_cost_map,
                request_batch, active_requests,
            )

            # result = self.optimizer.run(feature_scores, request_batch, active_requests)
            self.last_loss = None

            if self.config.REBALANCING:
                rebalancing_optimizer = CO_RebalancingCoverage(self.config)
                result = rebalancing_optimizer.run(result, vehicle_handler.vehicles, request_batch)
   
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
        assignment_status = {PayloadParser.STATS_ASSIGNED: result.request_assignment, 
                            PayloadParser.STATS_UNSERVED: list(unserved_requests)}
        data_logger.info("Status", extra={"timestamp": payload_object.current_time, "status": assignment_status})  

        return updated_driver_runs

    def _compute_fy_loss(
        self,
        feature_matrix: np.ndarray,
        ilp_model,
        x_t,
        trip_cost_count: int,
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
        """
        y_star = extract_y_binary(ilp_model, x_t, trip_cost_count)

        if y_star.sum().item() == 0:
            console_logger.warning(
                "FY loss skipped: CO_TripCostMinimization returned no feasible assignment."
            )
            self.last_loss = None
            return

        feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        scores = self.model(feature_tensor)     # (n,)

        oracle = make_map_oracle(
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
        console_logger.info(f"FY loss computed: {self.last_loss.item():.4f}")





