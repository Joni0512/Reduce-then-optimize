from enum import Enum
from rtv_solver.structure.node import Node

class VehicleStop:
    """
    VehicleStop combines the information of when, where and with which action a stop is performed and and how long it waits in the location to fulfill its action.
    """
    # TODO when code runs cleanly, turn stringLiterals to use Enum
    ACT_PICKUP = 'pickup'
    ACT_DROPOFF = 'dropoff'
    ACT_REBALANCE = 'rebalance'
    ACT_DEPOT = 'depot'
    ACT_UNKNOWN = 'unknown'

    def __init__(self, 
                 trip_id: str, 
                 node: Node, 
                 type: str, 
                 dwell: int):
        self.trip_id: int = trip_id
        self.node: Node = node
        assert type in [VehicleStop.ACT_PICKUP, VehicleStop.ACT_DROPOFF, VehicleStop.ACT_DEPOT, VehicleStop.ACT_REBALANCE]
        self.type: str = type
        self.dwell: int = dwell
        # NOTE all information below should be directly inferrable from the trip?
        self.stop_time: int = None
        self.request_id: int = None
        self.vehicle_id: int = None

    def set_request_id(self, rid):
        # TODO should be derived from trip directly? make use of it in the code; anything it should test
        assert self.request_id is None # for one stop, rid should not just be overwritten
        self.request_id = rid

    def __str__(self):
        return f"<VehicleStop - Trip {self.trip_id}, node: {self.node}, type: {self.type}, dwell: {self.dwell}>"
        
    def get_log(self):
        return "{0},{1},{2},{3},{4},{5}".format(self.node.lat, self.node.lon, self.type, self.stop_time, self.request_id, self.vehicle_id)
