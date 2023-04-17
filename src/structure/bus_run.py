class BusRun:
    def __init__(self, stops, arrival_times, departure_times):
        self.stops = stops
        self.arrival_times = arrival_times
        self.departure_times = departure_times
        self.load = [0]*len(stops)
        self.average_occupancy = 0
        self.max_occupancy = 0
