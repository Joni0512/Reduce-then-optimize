from rtv_solver.online_rtv_solver import OnlineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.config import Config

from rtv_solver.schema.payload_keys import PayloadKeys

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class OfflineRTVSolver:
    def __init__(self, config: Config):
        """container for the offline RTV solver that uses the online RTV solver in batches over a rolling horizon"""
        self.config = config
        # does not require multiprocessing scheme for MacoS as we do call OnlineRTVSolver as an object; it would break when being called for the second time

    def solve_rtv(self, payload, interval, step_size):
        online_rtv_solver = OnlineRTVSolver(self.config)
        # determine time interval of entire requests set
        start_time, end_time = PayloadParser.get_requests_time_interval(payload)
        # start before the initial start_time to catch all requests in the first interval
        current_time = max(0, start_time - interval)

        # track progress of solver iterations
        iteration = 0

        driver_runs = payload[PayloadKeys.DRIVERS]

        while current_time < end_time:
            console_logger.info(f"=== Iteration {iteration} offline RTV Solver at time {current_time} ===")
            
            # select requests that are to be considered in the current interval with pickup_window [current_time, current_time + interval]
            selected_requests = {}
            for request in payload[PayloadKeys.REQUESTS]:
                if ( # start of time window is part of current batch_interval
                    request[PayloadKeys.REQ_PICKUP_WINDOW_START] >= current_time
                    and 
                    request[PayloadKeys.REQ_PICKUP_WINDOW_START] < current_time + interval 
                    ):
                    selected_requests[request[PayloadKeys.REQ_BOOKING_ID]] = request
            
            # remove requests that are already part of vehicles; covered by PayloadParser in OnlineRTVsolver
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
                PayloadKeys.DRIVERS: driver_runs}

            if payload.get(PayloadKeys.TIME_MATRIX) is not None:
                new_payload[PayloadKeys.TIME_MATRIX] = payload[PayloadKeys.TIME_MATRIX]
            else:
                new_payload[PayloadKeys.TIME_MATRIX] = None

            # solve the RTV problem and update manifests
            if len(selected_requests) == 0:
                new_driver_runs, assignment_status = driver_runs, {PayloadKeys.STATS_ASSIGNED: {}, PayloadKeys.STATS_UNSERVED: []}
            else:    
                new_driver_runs, assignment_status = online_rtv_solver.solve_pdptw_rtv(new_payload, iteration)
            
            data_logger.info("Status", extra={"timestamp": current_time, "status": assignment_status})                
            # increment time (might not be the size of the batch) and iteration
            current_time += step_size 
            iteration += 1

            # update vehicles based on decisions in the previous step until current time (might not be the entire interval)
            tt_matrix = new_payload.get(PayloadKeys.TIME_MATRIX)
            simulated_driver_runs = OnlineRTVSolver.simulate_manifest(self.config, 
                                                                      current_time ,
                                                                      new_driver_runs,
                                                                      tt_matrix=tt_matrix)
            driver_runs = simulated_driver_runs

        final_driver_runs = OnlineRTVSolver.finalize_driverRuns(self.config, driver_runs, payload[PayloadKeys.DEPOT])
        
        return final_driver_runs
