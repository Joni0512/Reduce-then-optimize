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
    ACT_UNKNOWN = ''
    # TODO turn all dict comparisons into function of this class in order to handle less dictionary content

    def __init__(self, 
                 trip_id: str, 
                 node: Node, 
                 type: str, 
                 dwell: int):
        self.trip_id: str = trip_id
        self.node: Node = node # MANIFEST_LOC
        assert type in [VehicleStop.ACT_PICKUP, VehicleStop.ACT_DROPOFF, VehicleStop.ACT_DEPOT, VehicleStop.ACT_REBALANCE]
        self.type: str = type # MANIFEST_ACTION
        self.dwell: int = dwell # either pickup or alight
        
        # TODO define this information it would facilitate the information handling a lot
        # NOTE (test later) why do we never use this information, can we just remove them
        # self.stop_time: int = None          # MANIFEST_SCHED_TIME
        # self.request_id: int = None         # MANIFEST_BOOKING_ID
        # self.vehicle_id: int = None         # MANIFEST_RUN_ID
        # self.order: int = None              # MANIFEST_ORDER
        # self.ambulatory: int = None         # MANIFEST_AMBULATORY
        # self.wheelchair: int = None         # MANIFEST_WHEELCHAIR
        # self.time_window_start: int = None  # MANIFEST_TIME_WINDOW_START
        # self.time_window_end: int = None    # MANIFEST_TIME_WINDOW_END

    # TODO add create method from_dict or something similar that we can get directly from the dictionary and also move it back to the dictionary again (however we could have transitions and conditions that ensure the right changes)
    def set_request_id(self, rid):
        # TODO should be derived from trip directly? make use of it in the code; anything it should test
        assert self.request_id is None # for one stop, rid should not just be overwritten
        self.request_id = rid

    def __str__(self):
        return f"<VehicleStop - Trip {self.trip_id}, node: {self.node}, type: {self.type}, dwell: {self.dwell}>"
        
    def get_log(self):
        return "{0},{1},{2},{3},{4},{5}".format(self.node.lat, self.node.lon, self.type, self.stop_time, self.request_id, self.vehicle_id)
