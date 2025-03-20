from online_rtv_solver import OnlineRTVSolver
from handlers.request_handler import RequestHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler
from handlers.payload_parser import PayloadParser
import copy

class OfflineRTVSolver:

    def __init__(self,server_url,SHAREABLE_COST_FACTOR=1,RTV_TIMEOUT=3000, LARGEST_TSP = 10):
        self.ILP_SOLVER_TIMEOUT = 120 # seconds
        self.RTV_TIMEOUT = RTV_TIMEOUT #seconds
        self.PENALTY = 1000000 # penalty for not serving a trip
        self.SHAREABLE_COST_FACTOR = SHAREABLE_COST_FACTOR
        self.MAX_CARDINALITY = 4
        self.MAX_THREAD_CNT = 64
        self.REBALANCING = False
        self.RH_FACTOR = 1
        self.DWELL_PICKUP = 180
        self.DWELL_ALIGHT = 60
        self.LARGEST_TSP = LARGEST_TSP
        self.server_url = server_url

    def solve_rtv(self, payload, interval, step_size):
        online_rtv_solver = OnlineRTVSolver(self.server_url)

        # get the start_time
        start_time = 24*3600
        end_time = 0

        for request in payload["requests"]:
            if request["pickup_time_window_start"] < start_time:
                start_time = request["pickup_time_window_start"]
            if request["dropoff_time_window_end"] > end_time:
                end_time = request["dropoff_time_window_end"]

        current_time = max(0,start_time - interval)
        driver_runs = payload["driver_runs"]

        while current_time < end_time:
            # select the requests that are to be considered in the current interval
            selected_requests = {}
            for request in payload["requests"]:
                if request["pickup_time_window_start"] < current_time + interval and request["pickup_time_window_start"] >= current_time:
                    selected_requests[request["booking_id"]] = request

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

            # solve the RTV problem
            if len(selected_requests) == 0:
                new_driver_runs = driver_runs
            else:             
                new_driver_runs = online_rtv_solver.solve_rtv(current_time,new_payload)
            current_time += step_size

            # simulate the driver runs
            simulated_driver_runs = online_rtv_solver.simulate_manifest(current_time,new_driver_runs,intermediate_location=False)
            driver_runs = simulated_driver_runs

        return driver_runs
