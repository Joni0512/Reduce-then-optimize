from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtv_solver.structure.vehicle import Vehicle
    from rtv_solver.structure.request import Request    
    from rtv_solver.structure.node import Node

class Trip:
    """
    Trip-related information
    TODO typing?, 
    NOTE cost double? <> relation to TripCost?
    """
    def __init__(
            self, 
            request_id: int, 
            trip_number: int, 
            am_capacity: int, 
            wc_capacity: int, 
            pick_up_time: float, 
            latest_pick_up_time: float, 
            earliest_arrival_time: float,
            latest_arrival_time: float, 
            origin: Node, 
            destination: Node, 
            dwell_pickup: int, 
            dwell_alight: int, 
            iteration: int, 
            cost: float = None, 
            bus_combination = None, 
            first_last_mile_type: int = 0, 
            vehicle: Vehicle = None):
        # identifiers
        self.request_id = request_id
        self.number = trip_number
        self.iteration = iteration
        if bus_combination == None:
            self.id = "{0}-{1}".format(iteration, request_id)
        else:
            self.id = "{0}:{1}-{2}".format(request_id, bus_combination, first_last_mile_type)
        if vehicle is not None:
            self.vehicle = vehicle # NOTE seems to not be used ever
        # Location information 
        self.origin = origin
        self.destination = destination
        # Conditions for timing
        self.pick_up_time = pick_up_time
        self.latest_pick_up_time = latest_pick_up_time
        self.earliest_arrival_time = earliest_arrival_time
        self.latest_arrival_time = latest_arrival_time
        # Metadata of the trip
        self.cost = cost
        self.dwell_pickup = dwell_pickup
        self.dwell_alight = dwell_alight
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
        self.bus_combination = bus_combination # what is the bus used for?
        self.first_last_mile_type = first_last_mile_type
        # simulation state
        self.picked = False
        self.shared_trips = {}

    @classmethod
    def from_request(cls, trip_number, request: Request, iteration, cost = None, bus_combination = None, first_last_mile_type = 0) -> Trip:
        return Trip(
            request.id,
            trip_number,
            request.am_capacity,
            request.wc_capacity,
            request.earliest_pickup_time,
            request.latest_pickup_time,
            request.earliest_arrival_time,
            request.latest_arrival_time,
            request.origin,
            request.destination,
            request.dwell_pickup,
            request.dwell_alight,
            iteration,
            cost,
            bus_combination, 
            first_last_mile_type
        )

    def __str__(self):
        return f"<Trip {self.id}: pickup: {self.pick_up_time}, origin: {self.origin}, destination: {self.destination}>"
        
    def __repr__(self):
        return (
            f"Trip("
            f"request_id={self.request_id!r}, "
            f"trip_number={self.number!r}, "
            f"am_capacity={self.am_capacity!r}, "
            f"wc_capacity={self.wc_capacity!r}, "
            f"pick_up_time={self.pick_up_time!r}, "
            f"latest_pick_up_time={self.latest_pick_up_time!r}, "
            f"earliest_arrival_time={self.earliest_arrival_time!r}, "
            f"latest_arrival_time={self.latest_arrival_time!r}, "
            f"origin={self.origin!r}, "
            f"destination={self.destination!r}, "
            f"dwell_pickup={self.dwell_pickup!r}, "
            f"dwell_alight={self.dwell_alight!r}, "
            f"iteration={self.iteration!r}, "
            f")"
        )

    # NOTE: all methods below never used
    def get_shared_trips(self):
        trips = []
        for cardinality in self.shared_trips:
            trips.extend(self.shared_trips[cardinality])
        return trips
    
    def get_shared_trips_of_cardinality(self, cardinality):
        return self.shared_trips[cardinality]

    def add_shared_trip(self, cardinality, trip_id):
        if cardinality not in self.shared_trips:
            self.shared_trips[cardinality] = []
        self.shared_trips[cardinality].append(trip_id)
