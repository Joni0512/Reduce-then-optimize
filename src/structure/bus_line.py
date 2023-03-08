class BusLine:
    def __init__(self, bus_line, frequency, stops, travel_time, no_of_busses, bus_waiting_time):
        self.bus_line = bus_line
        self.frequency = frequency
        self.stops = stops
        self.travel_time = travel_time
        self.no_of_busses = no_of_busses
        self.bus_waiting_time = bus_waiting_time
        self.busses = []

    def __str__(self):
        return "{{Bus line: {0}, frequency: {1}, travel time: {2} mins, fleet size: {3}, wait time: {4} mins}}".format(self.bus_line,self.frequency,self.travel_time/60, self.no_of_busses, self.bus_waiting_time/60)
