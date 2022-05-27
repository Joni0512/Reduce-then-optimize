class SharedTrip:
    def __init__(self, number, trips,cost):
        self.number = number
        self.trips = trips
        self.cardinality = len(trips)
        self.cost = cost
