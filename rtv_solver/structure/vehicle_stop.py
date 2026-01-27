from enum import Enum
from rtv_solver.structure.node import Node

class VehicleStop:
    """
    VehicleStop combines the information of when, where and with which action a stop is performed and and how long it waits in the location to fulfill its action.
    """
    # TODO when code runs cleanly, turn stringLiterals to use Enum
    PICKUP = 'pickup'
    DROPOFF = 'dropoff'
    REBALANCE = 'rebalance'
    DEPOT = 'depot'
    UNKNOWN = 'unknown'

    def __init__(self, 
                 trip_id: str, 
                 node: Node, 
                 type: int, 
                 dwell: int):
        self.trip_id: int = trip_id
        self.node: Node = node
        self.type: str = type # TODO change to StopType Enum for easier handling
        self.dwell: int = dwell
        # TODO all information below should be directly inferrable from the trip?
        self.stop_time: int = None
        self.request_id: int = None
        self.vehicle_id: int = None

    def set_request_id(self, rid):
        # TODO should be derived from trip directly? make use of it in the code; anything it should test
        assert self.request_id is None # for one stop, rid should not just be overwritten
        self.request_id = rid

    def __str__(self):
        return f"<VehicleStop - Trip {self.trip_id}, node: {self.node}, type: {self.translate_type()}, dwell: {self.dwell}>"

    def translate_type(self):
        type_name = "UNKNOWN"
        if self.type == 0:
            type_name = "PICKUP"
        if self.type == 1:
            type_name = "DROPOFF"
        elif self.type == 2:
            type_name = "REBALANCE"
        elif self.type == 3:
            type_name = "DEPOT"
        assert type_name != "UNKNOWN", "Invalid stop type"
        return type_name
        
    def get_log(self):
        type_name = self.translate_type()
        return "{0},{1},{2},{3},{4},{5}".format(self.node.lat, self.node.lon, type_name, self.stop_time, self.request_id, self.vehicle_id)
