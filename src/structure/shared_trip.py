class SharedTrip:
    def __init__(self, number, trips,cost,sequence):
        self.number = number
        self.trips = trips
        self.cardinality = len(trips)
        self.cost = cost
        self.sequence = sequence
