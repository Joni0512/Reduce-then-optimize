class VehicleStop:
    def __init__(self, trip_id, node, type):
        self.trip_id = trip_id
        self.node = node
        self.type = type

    def __str__(self):
        return "{{Trip ID: {0}, node: {1}, type: {2}}}".format(self.trip_id, self.node, self.type)
