from enum import Enum

# TODO when code runs cleanly, turn conditionals to use Enum
class StopType(Enum):
    PICKUP = 0
    DROPOFF = 1
    REBALANCE = 2
    DEPOT = 3

class VehicleStop:
    """
    Docstring for VehicleStop
    """
    def __init__(self, trip_id, node, type, dwell):
        self.trip_id = trip_id
        self.node = node
        self.type = type
        self.stop_time = None
        self.request_id = None
        self.vehicle_id = None
        self.dwell = dwell

    def __str__(self):
        return "{{Trip ID: {0}, node: {1}, type: {2}}}".format(self.trip_id, self.node, self.type)

    def get_log(self):
        type_name = "PICKUP"
        if self.type == 1:
            type_name = "DROPOFF"
        elif self.type == 2:
            type_name = "REBALANCE"
        elif self.type == 3:
            type_name = "DEPOT"
        return "{0},{1},{2},{3},{4},{5}".format(self.node.lat,self.node.lon,type_name,self.stop_time,self.request_id,self.vehicle_id)
