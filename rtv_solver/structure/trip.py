from __future__ import annotations

from rtv_solver.structure.request import Request

class Trip:
    """
    Trip-related information
    TODO typing?, 
    NOTE cost double? <> relation to TripCost?
    """
    def __init__(
            self, 
            request_id, 
            trip_number, 
            am_capacity, 
            wc_capacity, 
            pick_up_time, 
            latest_pick_up_time, 
            earliest_arrival_time,
            latest_arrival_time, 
            origin, 
            destination, 
            dwell_pickup, 
            dwell_alight, 
            iteration, 
            cost = None, 
            bus_combination = None, 
            first_last_mile_type = 0, 
            vehicle = None):
        # identifiers
        self.request_id: int = request_id
        self.number: int = trip_number
        if bus_combination == None:
            self.id = "{0}-{1}".format(iteration, request_id)
        else:
            self.id = "{0}:{1}-{2}".format(request_id, bus_combination, first_last_mile_type)
        if vehicle is not None:
            self.vehicle = vehicle
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
            request.pick_up_time,
            request.latest_pick_up_time,
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

    # NOTE DEBUG Checks
    @property
    def vehicle(self):
        print(f"Trip {self.id}: vehicle accessed")
        return self._vehicle

    @vehicle.setter
    def vehicle(self, vehicle):
        if vehicle is None:
            raise ValueError("vehicle cannot be None")
        if self._vehicle is not None:
            raise RuntimeError("Vehicle already assigned")
        self._vehicle = vehicle

    def __str__(self):
        return f"<Trip {self.id}: pickup: {self.pick_up_time}, origin: {self.origin}, destination: {self.destination}>"

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
