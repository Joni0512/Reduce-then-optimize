class Trip:
    def __init__(self, id, request_id, number, pick_up_time, arrival_time, origin, destination, cost, bus_combination = None, vehicle=None):
        self.origin = origin
        self.destination = destination
        self.pick_up_time = pick_up_time
        self.id = id
        self.number = number
        self.arrival_time = arrival_time
        self.bus_combination = bus_combination
        self.vehicle = vehicle
        self.picked = False
        self.shared_trips = {}
        self.request_id = request_id
        self.cost = cost

    def __str__(self):
        return "{{ID: {0}, time: {1}, origin: {2}, destination: {3}}}".format(self.id,self.pick_up_time,self.origin,self.destination)

    def get_shared_trips(self):
        trips = []
        for cardinality in self.shared_trips:
            trips.extend(self.shared_trips[cardinality])
        return trips
    
    def get_shared_trips_of_cardinality(self,cardinality):
        return self.shared_trips[cardinality]

    def add_shared_trip(self,cardinality,trip_id):
        if cardinality not in self.shared_trips:
            self.shared_trips[cardinality] = []
        self.shared_trips[cardinality].append(trip_id)
