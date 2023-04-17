class BusLine:
    def __init__(self, bus_line,stops,capacity):
        self.bus_line = bus_line
        self.stops = stops
        self.bus_runs = []
        self.capacity = capacity
        self.first_incomplete_run = 0

    def __str__(self):
        return "{{Bus line: {0}}}".format(self.bus_line)