class Request:
    def __init__(self, 
                 identifier, 
                 am_capacity, 
                 wc_capacity, 
                 pick_up_time, 
                 latest_pick_up_time, 
                 earliest_arrival_time,
                 latest_arrival_time, 
                 origin, 
                 destination,
                 dwell_pickup,
                 dwell_alight,
                 priority=1):
        self.origin = origin
        self.destination = destination
        self.pick_up_time = pick_up_time
        self.latest_pick_up_time = latest_pick_up_time
        self.id = identifier
        self.earliest_arrival_time = earliest_arrival_time
        self.latest_arrival_time = latest_arrival_time
        self.dwell_pickup = dwell_pickup
        self.dwell_alight = dwell_alight
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
        self.priority = priority

    def __str__(self):
        return f"<Request {self.id}:, pickup: {self.pick_up_time}, origin: {self.origin}, destination: {self.destination}, priority: {self.priority}>"
