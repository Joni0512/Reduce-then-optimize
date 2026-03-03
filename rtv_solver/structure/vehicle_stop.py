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
        """
        TODO In order to reduce calls to the network handler backend server, the travel time can be passed as a parameter and be used in the feature builder.
        travel_time: float = travel_time from the last vehicle position to the position of this stop (only used for feature creation in COAML pipeline)
        """
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

    def get_requestID_stripped_iteration(self):
        """normal structure of trip_id is 'iteration-request_id' """
        return self.trip_id.split("-")[1]

    def __str__(self):
        return f"<VehicleStop - Trip {self.trip_id}, node: {self.node}, type: {self.type}, dwell: {self.dwell}>"
    
    def __repr__(self):
        return (
            f"VehicleStop("
            f"trip_id={self.trip_id!r}, "
            f"node={self.node!r}, "
            f"type={self.type!r}, "
            f"dwell={self.dwell!r}"
            f")"
        )
