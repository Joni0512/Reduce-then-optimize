class SharedTrip:
    """
    TODO Docstring for ShareTrip and when it is used
    """
    def __init__(self, prev_trip_number, number, trips,cost,sequence):
        self.number = number # initially 0 and overwritten before it is actually tested
        self.trips = trips
        self.cardinality = len(trips)
        self.cost = cost
        self.sequence = sequence
        self.prev_trip_number = prev_trip_number

    def __str__(self):
        return f"<SharedTrip {self.number}: cardinality: {self.cardinality}, cost: {self.cost}, trips: {self.trips}, sequence: {self.sequence}>"
    
    def set_number(self, number):
        print("Setting shared trip number from {0} to {1}".format(self.number,number)) # testing purposes how often this really works, TODO remove
        self.number = number
