import logging

from rtv_solver.online_rtv_solver import OnlineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.config import Config

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

        unserved_requests = []
        driver_runs = payload[PayloadParser.DRIVERS]

        while current_time < end_time:
            logging.info(f"Iteration {iteration} offline RTV Solver at time {current_time}")
            
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
                new_driver_runs = driver_runs
            else:    
                new_driver_runs, unserved = online_rtv_solver.solve_pdptw_rtv(new_payload, iteration)
                unserved_requests.extend(unserved)
                
            # increment time (might not be the size of the batch) and iteration
            current_time += step_size 
            iteration += 1

            # update vehicles based on decisions in the previous step until current time (might not be the entire interval)
            # FIXME currently it never updates the manifest (JW requests need to be removed from manifest if they have not been picked up yet as in the next iteration we want to reoptimize based on the last position and not a fixed schedule for the next how many steps)
            # TODO this fix functionality already exists in simulate_manifest_new(...), but first we need to fix the general rolling horizon issue or they might be related but it probably requires more than just my fix
            simulated_driver_runs = online_rtv_solver.simulate_manifest(current_time , new_driver_runs, intermediate_location=True)
            driver_runs = simulated_driver_runs

            feasible, stats = online_rtv_solver.get_stats(depot=payload[PayloadParser.DEPOT], driver_runs=driver_runs)
            logging.info(f"Original stats: {stats}")

        # TODO unserved_requests is incorrect, some parts are recounted
        return driver_runs, unserved_requests
