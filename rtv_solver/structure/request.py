from rtv_solver.structure.node import Node
from rtv_solver.schema.payload_keys import PayloadKeys


class Request:
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"identifier={self.id!r}, "
            f"earliest_pickup_time={self.earliest_pickup_time!r}, "
            f"latest_pickup_time={self.latest_pickup_time!r}, "
            f"earliest_arrival_time={self.earliest_arrival_time!r}, "
            f"latest_arrival_time={self.latest_arrival_time!r}, "
            f"origin={self.origin!r}, "
            f"destination={self.destination!r}, "
            f"dwell_pickup={self.dwell_pickup!r}, "
            f"dwell_alight={self.dwell_alight!r}, "
            f"am_capacity={self.am_capacity!r}, "
            f"wc_capacity={self.wc_capacity!r}, "
            f"priority={self.priority!r}"
            f")"
        )
    
    @staticmethod
    def resolve_dwell_times(
        req: dict,
        default_pickup: int = 180,
        default_alight: int = 60,
    ) -> tuple[int, int]:
        """
        Resolve dwell pickup and dwell alight from request payload or defaults.
        Priority: "pickup_service_time"/"dropoff_service_time" (inital transfer from LiLimParser) > REQ_DWELL_PICKUP/REQ_DWELL_ALIGHT > defaults.
        """
        pk = PayloadKeys
        if "pickup_service_time" in req and "dropoff_service_time" in req:
            return req["pickup_service_time"], req["dropoff_service_time"]
        if pk.REQ_DWELL_PICKUP in req and pk.REQ_DWELL_ALIGHT in req:
            return req[pk.REQ_DWELL_PICKUP], req[pk.REQ_DWELL_ALIGHT]
        return default_pickup, default_alight

    def get_direct_travel_time(self):
        return self.earliest_arrival_time - self.earliest_pickup_time

    def get_time_window_duration(self):
        pickup_duration = float(self.latest_pickup_time - self.earliest_pickup_time)
        dropoff_duration = float(self.latest_arrival_time - self.earliest_arrival_time)
        if abs( pickup_duration - dropoff_duration ) > 0.00001:
            print(f"DEBUG {self.id}: pickup_duration: {pickup_duration}, dropoff_duration: {dropoff_duration}")
        return dropoff_duration
