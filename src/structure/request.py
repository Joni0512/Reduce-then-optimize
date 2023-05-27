class Request:
    def __init__(self, id, pick_up_time, latest_pick_up_time, arrival_time, origin, destination,dwell_pickup,dwell_alight):
        self.origin = origin
        self.destination = destination
        self.pick_up_time = pick_up_time
        self.latest_pick_up_time = latest_pick_up_time
        self.id = id
        self.arrival_time = arrival_time
        self.dwell_pickup = dwell_pickup
        self.dwell_alight = dwell_alight

    def __str__(self):
        return "{{ID: {0}, time: {1}, origin: {2}, destination: {3}}}".format(self.id,self.pick_up_time,self.origin,self.destination)
