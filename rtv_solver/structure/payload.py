from rtv_solver.structure.node import Node

class Payload:
    """
    Parsed Payload object from the JSON response from OSRM backend server.
    """
    def __init__(self, travel_time_matrix, current_time, requests, boarded_requests_keys, active_requests_keys, driver_runs, depot: Node):
        self.travel_time_matrix = travel_time_matrix
        self.current_time: int = current_time # only used for rebalancing trip and those vehicles must be idling, so it should be the same time as the iteration
        self.requests: list[dict[str, str]] = requests
        self.boarded_requests_keys: list[str] = boarded_requests_keys
        self.active_requests_keys: list[str] = active_requests_keys
        self.driver_runs: list[dict[str, str]] = driver_runs
        self.depot: Node = depot

    def __str__(self):
        return f"<Payload: travel_time_matrix: {self.travel_time_matrix}, current_time: {self.current_time}, requests: {self.requests}, boarded_request_keys: {self.boarded_requests_keys}, active_request_keys: {self.active_requests_keys}, driver_runs: {self.driver_runs}, depot: {self.depot}>"
