class Vehicle:
    def __init__(self, id, start_node, capacity, start_time):
        self.id = id
        self.start_time = start_time
        self.time_at_last = start_time
        self.time_at_next = start_time
        self.capacity = capacity
        self.trips = {}
        self.picked = []
        self.served_trips = []
        self.last_node = start_node
        self.stop_sequence = []
        self.started = False
        self.rebalancing = False

    def __str__(self):
        return "{{Vehicle ID: {0}, capacity: {1}, start node: {2}, start time: {3}}}".format(self.id, self.capacity, self.next_node, self.start_time)
