from rtv_solver.structure.node import Node

class Request:
    # TODO add type hints
    def __init__(self, 
                 identifier: int, 
                 earliest_pickup_time: float, 
                 latest_pickup_time: float, 
                 earliest_arrival_time: float,
                 latest_arrival_time: float, 
                 origin: Node, 
                 destination: Node,
                 dwell_pickup: int,
                 dwell_alight: int,
                 am_capacity: int, 
                 wc_capacity: int, 
                 priority: int =1):
        self.id = identifier
        # location and time information
        self.origin = origin
        self.destination = destination
        self.earliest_pickup_time = earliest_pickup_time
        self.latest_pickup_time = latest_pickup_time
        self.earliest_arrival_time = earliest_arrival_time
        self.latest_arrival_time = latest_arrival_time
        # request metadata
        self.dwell_pickup = dwell_pickup
        self.dwell_alight = dwell_alight
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
        self.priority = priority

    def __str__(self):
        return f"<Request {self.id}:, pickup: {self.earliest_pickup_time}, origin: {self.origin}, destination: {self.destination}, priority: {self.priority}>"
