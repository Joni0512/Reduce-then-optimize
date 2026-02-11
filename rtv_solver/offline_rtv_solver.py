from rtv_solver.online_rtv_solver import OnlineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.config import Config

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class OfflineRTVSolver:
    def __init__(self, config: Config):
        """container for the offline RTV solver that uses the online RTV solver in batches over a rolling horizon"""
        self.config = config

    def solve_rtv(self, payload, interval, step_size):
        online_rtv_solver = OnlineRTVSolver(self.config)
        # determine time interval of entire requests set
        start_time, end_time = PayloadParser.get_requests_time_interval(payload)
        # start before the initial start_time to catch all requests in the first interval
        current_time = max(0, start_time - interval)

        # track progress of solver iterations
        iteration = 0

        driver_runs = payload[PayloadParser.DRIVERS]

        while current_time < end_time:
            console_logger.info(f"=== Iteration {iteration} offline RTV Solver at time {current_time} ===")
            
            # select requests that are to be considered in the current interval with pickup_window [current_time, current_time + interval]
            selected_requests = {}
            for request in payload[PayloadParser.REQUESTS]:
                if ( # start of time window is part of current batch_interval
                    request[PayloadParser.REQ_PICKUP_WINDOW_START] >= current_time
                    and 
                    request[PayloadParser.REQ_PICKUP_WINDOW_START] < current_time + interval 
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
                PayloadParser.DRIVERS: driver_runs}

            # solve the RTV problem and update manifests
            if len(selected_requests) == 0:
                new_driver_runs, assignment_status = driver_runs, {PayloadParser.STATS_ASSIGNED: {}, PayloadParser.STATS_UNSERVED: []}
            else:    
                new_driver_runs, assignment_status = online_rtv_solver.solve_pdptw_rtv(new_payload, iteration)
            
            # TODO i want the status development of requests (active, boarded, unserved, delivered) 
            # TODO how do I get the status for already delivered requests
            data_logger.info("Status", extra={"timestamp": current_time, "status": assignment_status})                
            # increment time (might not be the size of the batch) and iteration
            current_time += step_size 
            iteration += 1

            # update vehicles based on decisions in the previous step until current time (might not be the entire interval)
            simulated_driver_runs = OnlineRTVSolver.simulate_manifest(self.config, 
                                                                      current_time ,
                                                                      new_driver_runs, 
                                                                      intermediate_location=True)
            driver_runs = simulated_driver_runs

        final_driver_runs = OnlineRTVSolver.finalize_driverRuns(driver_runs, payload[PayloadParser.DEPOT])
        # TODO update assignment_devlopment calculation. based on the data stores to JSONL instead of handing it over here
        return final_driver_runs
