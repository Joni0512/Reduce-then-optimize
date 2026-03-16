from rtv_solver.structure.trip import Trip

class SharedTrip:
    def __init__(self, prev_trip_number: int, trip_number: int, trips: set[Trip], cost: float, sequence: list):
        self.number = trip_number # initially 0 and overwritten before it is actually tested
        self.trips = trips
        self.cardinality = len(trips)
        self.cost = cost
        self.sequence = sequence
        self.prev_trip_number = prev_trip_number

    def __str__(self):
        return f"<SharedTrip {self.number}: cardinality: {self.cardinality}, cost: {self.cost}, trips: {self.trips}, sequence: {self.sequence}>"
    def __repr__(self):
        return (
            f"SharedTrip("
            f"prev_trip_number={self.prev_trip_number!r}, "
            f"trip_number={self.number!r}, "
            f"trips={self.trips!r}, "
            f"cost={self.cost!r}, "
            f"sequence={self.sequence!r}"
            f")"
        )
