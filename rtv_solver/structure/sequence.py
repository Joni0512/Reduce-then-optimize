from rtv_solver.structure.vehicle_stop import VehicleStop

class StopSequence(list):
    """
    List-like container for a sequence of VehicleStops:
    1. Each trip_id occurs exactly twice.
    2. Each trip_id has one pickup and one dropoff.
    3. Pickup occurs before dropoff in the list order."""
    
    def __init__(self, stops: list[VehicleStop]):
        # initialize directly from the list; so that append works as well
        super().__init__(stops)
        self._validate()

    def _validate(self):
        """checks whether the sequence is valid (each pickup is also dropped off in a feasible order)"""
        # TODO currently wrong - DROPOFFs without PICKUPs are possible if we have not yet arrived
        trips = {}

        for index, stop in enumerate(self):
            trip_id = stop.trip_id
            stop_type = stop.type

            if trip_id not in trips:
                trips[trip_id] = []

            trips[trip_id].append((stop_type, index))

        for trip_id, entries in trips.items():
            if len(entries) != 2:
                raise ValueError(f"Trip {trip_id} must have exactly 2 stops, got {len(entries)}")
            (type_1, idx_1), (type_2, idx_2) = entries
            if type_1 == type_2:
                raise ValueError(f"Trip {trip_id} must have one pickup and one dropoff, got {type_1} twice")
            if (type_1 == VehicleStop.ACT_PICKUP and idx_1 > idx_2) or (type_2 == VehicleStop.ACT_PICKUP and idx_2 > idx_1):
                raise ValueError(f"Trip {trip_id} pickup must occur before dropoff")
            
    def __str__(self):
        """print out the sequence in order to make it readable in-line"""
        seq_str = "\n \t" + "\n\t".join([str(node) for node in self.nodes])
        return seq_str


