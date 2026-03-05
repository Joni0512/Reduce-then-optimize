from collections.abc import Iterable

from rtv_solver.structure.vehicle_stop import VehicleStop

class StopSequence(list[VehicleStop]):
    """
    List-like container for a sequence of VehicleStops:
    1. Each trip_id occurs exactly twice.
    2. Each trip_id has one pickup and one dropoff.
    3. Pickup occurs before dropoff in the list order.
    """
    
    def __init__(self, stops: Iterable[VehicleStop]):
        # initialize directly from the list; so that append works as well
        super().__init__(stops)
        self._validate()

    def _validate(self):
        """
        Validate stop sequence.
        Allowed cases per trip_id:
        - [PICKUP, DROPOFF]  -> normal
        - [DROPOFF]          -> allowed (pickup happened in previous sequence earlier > vehicle already carrying)
        Disallowed:
        - more than 2 stops for same trip_id
        - two pickups or two dropoffs
        - dropoff before pickup when both exist
        """
        trips = {}

        for idx, stop in enumerate(self):
            trip_id = stop.trip_id
            stop_type = stop.type
            if trip_id not in trips:
                trips[trip_id] = []
            trips[trip_id].append((stop_type, idx))

        for trip_id, entries in trips.items():
            if len(entries) > 2:
                raise ValueError(f"Trip {trip_id} must not appear more than twice, got {len(entries)}")
            if len(entries) == 1:
                stop_type, _ = entries[0]
                if stop_type == VehicleStop.ACT_PICKUP:
                    raise ValueError(f"Trip {trip_id} cannot appear as PICKUP-only in the sequence")
                if stop_type != VehicleStop.ACT_DROPOFF:
                    raise ValueError(f"Trip {trip_id} has invalid stop type {stop_type}") # if it occurs, we should check for now
                continue
            
            (type_1, idx_1), (type_2, idx_2) = entries
            if type_1 == type_2:
                raise ValueError(f"Trip {trip_id} must have one pickup and one dropoff, got {type_1} twice")
            pickup_idx = idx_1 if type_1 == VehicleStop.ACT_PICKUP else idx_2
            dropoff_idx = idx_1 if type_1 == VehicleStop.ACT_DROPOFF else idx_2
            if pickup_idx > dropoff_idx:
                raise ValueError(f"Trip {trip_id} pickup must occur before dropoff")

    @staticmethod
    def sequence_to_string(sequence_list):
        """print out the sequence in order to make it readable in-line"""
        # staticMethod makes it more usable in other locations
        seq_str = "\n \t" + "\n\t".join([str(node) for node in sequence_list])
        return seq_str

    def simple_str(self):
        """get just the trip-IDs once of the sequence"""
        single_trip_ids = dict.fromkeys(stop.trip_id for stop in self)
        return ", ".join(single_trip_ids)

    def request_str(self):
        """get just the request-IDs once of the sequence"""
        single_request_ids = dict.fromkeys(stop.get_requestID_stripped_iteration() for stop in self)
        return "-".join(str(request_id) for request_id in single_request_ids)

    def __str__(self):
        return str(self.sequence_to_string(self))


