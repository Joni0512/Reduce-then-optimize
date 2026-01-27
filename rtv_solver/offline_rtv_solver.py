
from rtv_solver.online_rtv_solver import OnlineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser

class OfflineRTVSolver:
    def __init__(self, config):
        """container for the offline RTV solver that uses the online RTV solver in batches over a rolling horizon"""
        self.config = config

    def solve_rtv(self, payload, interval, step_size):
        online_rtv_solver = OnlineRTVSolver(self.config)
        # determine time interval of entire requests set
        start_time, end_time = PayloadParser.get_requests_time_interval(payload)
        # start before the initial start_time to catch all requests in the first interval
        current_time = max(0,start_time - interval)

        # track progress of the solver iterations
        iteration = 0

        unserved_requests = []
        driver_runs = payload["driver_runs"]

        while current_time < end_time:
            print("=== Offline RTV Solver Iteration", iteration, "at time", current_time, "===")
            
            # select requests that are to be considered in the current interval
            selected_requests = {}
            for request in payload["requests"]:
                if request["pickup_time_window_start"] < current_time + interval and request["pickup_time_window_start"] >= current_time:
                    selected_requests[request["booking_id"]] = request
            
            # remove requests that are already part of vehicles (NOTE covered through manifests in OnlineSolver?)
            for dr in driver_runs:
                for stop in dr["manifest"]:
                    if stop["booking_id"] in selected_requests:
                        del selected_requests[stop["booking_id"]]
            selected_requests = list(selected_requests.values())

            # create a new payload with the selected requests
            new_payload = {
                "depot": payload["depot"],
                "requests": selected_requests,
                "driver_runs": driver_runs}

            # solve the RTV problem and update manifests
            if len(selected_requests) == 0:
                new_driver_runs = driver_runs
            else:             
                new_driver_runs, unserved = online_rtv_solver.solve_pdptw_rtv(new_payload)
                unserved_requests.extend(unserved)
            current_time += step_size # take step, possibly only a partial increment compared to the interval size
            iteration += 1

            # update vehicles based on decisions in the previous step until current time
            simulated_driver_runs = online_rtv_solver.simulate_manifest(current_time, new_driver_runs, intermediate_location=False)
            driver_runs = simulated_driver_runs

        return driver_runs, unserved_requests
