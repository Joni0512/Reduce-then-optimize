from rtv_solver.structure.node import Node

class Payload:
    """
    Parsed Payload object from the JSON response from OSRM backend server.
    """
    def __init__(self, travel_time_matrix, current_time, requests, boarded_requests, active_requests, driver_runs, depot):
        self.travel_time_matrix = travel_time_matrix
        self.current_time: int = current_time
        self.requests: list[dict] = requests
        self.boarded_requests: list[dict] = boarded_requests
        self.active_requests: list[dict] = active_requests
        self.driver_runs: list[dict] = driver_runs
        self.depot: Node = depot

    def __str__(self):
        return f"<Payload: travel_time_matrix: {self.travel_time_matrix}, current_time: {self.current_time}, requests: {self.requests}, boarded_requests: {self.boarded_requests}, active_requests: {self.active_requests}, driver_runs: {self.driver_runs}, depot: {self.depot}>"
