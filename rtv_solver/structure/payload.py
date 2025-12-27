class Payload:
    """
    Parsed Payload object from the JSON response from OSRM backend server.
    """
    def __init__(self, travel_time_matrix, current_time, requests, boarded_requests, active_requests, driver_runs, depot):
        self.travel_time_matrix = travel_time_matrix
        self.current_time = current_time
        self.requests = requests
        self.boarded_requests = boarded_requests
        self.active_requests = active_requests
        self.driver_runs = driver_runs
        self.depot = depot

    def __str__(self):
        return "<Payload: {travel_time_matrix: {0}, current_time: {1}, requests: {2}, boarded_requests: {3}, active_requests: {4}, driver_runs: {5}, depot: {6}}>".format(self.travel_time_matrix, self.current_time, self.requests, self.boarded_requests, self.active_requests, self.driver_runs, self.depot)
