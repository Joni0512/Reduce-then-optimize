from enum import Enum

class VehicleStatus(Enum):
    # TODO add status instead of deleting vehicles when they are not active anymore; with an enum makes it easier
    ACTIVE = 1 # during operation
    INACTIVE = 2 # before start while in depot
    COMPLETED = 3 # after end
    
class Vehicle:
    """
    Vehicle-related information covering basic vehicle information and simulation state during runtime

    TODO differentiate initialized information and dynamic information to keep track of state
    """
    def __init__(self, id, start_node, am_capacity, wc_capacity, start_time, end_time, depot):
        # Static vehicle information
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.depot = depot
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
        
        # tracks simulation state
        self.started = False
        self.rebalancing = False
        self.dwelling = False
        self.time_at_last = start_time
        self.time_at_next = start_time
        self.trips = {}
        self.picked = []
        self.served_trips = []
        self.last_node = start_node
        self.stop_sequence = []
        self.final_stop_time = start_time

    def __str__(self):
        return f"Vehicle ID {self.id}: time: {self.start_time}>{self.end_time}, trips {self.trips}, picked {self.picked}"
