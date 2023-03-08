class BusRun:
    def __init__(self, intial_load, starting_time):
        self.intial_load = intial_load
        self.starting_time = starting_time
        self.load = {}
        self.average_occupancy = 0
        self.max_occupancy = 0
