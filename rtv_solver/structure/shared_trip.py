class SharedTrip:
    """
    TODO Docstring for ShareTrip and when it is used
    """
    def __init__(self, prev_trip_number, number, trips,cost,sequence):
        self.number = number # is this just an ID? always 0
        self.trips = trips
        self.cardinality = len(trips)
        self.cost = cost
        self.sequence = sequence
        self.prev_trip_number = prev_trip_number

    def __str__(self):
        return f"<SharedTrip {self.number}: cardinality: {self.cardinality}, cost: {self.cost}, trips: {self.trips}, sequence: {self.sequence}>"
