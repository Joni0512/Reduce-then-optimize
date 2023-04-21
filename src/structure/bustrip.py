class BusTrip:
    def __init__(self, bus_line, bus_run, pick_up_stop, pick_up_stop_node, destination_stop, destination_stop_node, leaving_time, arrival_time, transfer_point = -1, transfer_point_node = None, bus_line2= None, bus_run2 = None, arrival_at_transfer = None, departure_at_transfer=None):
        self.bus_lines = [bus_line]
        self.pick_up_stop = pick_up_stop
        self.pick_up_stop_node = pick_up_stop_node
        self.destination_stop = destination_stop
        self.destination_stop_node = destination_stop_node
        self.leaving_time = leaving_time
        self.arrival_time = arrival_time
        self.bus_run = [bus_run]
        if bus_line2 != None:
            self.bus_lines.append(bus_line2)
            self.bus_run.append(bus_run2)
        self.transfer_point = transfer_point
        self.transfer_point_node = transfer_point_node
        self.first_mile_trip = 0
        self.first_mile_trip_empty = False
        self.last_mile_trip = 0
        self.last_mile_trip_empty = False
        self.id = ",".join(self.bus_lines)
        self.arrival_at_transfer = arrival_at_transfer
        self.departure_at_transfer = departure_at_transfer

    # def __str__(self):
    #     return "{{Bus line: {0}, frequency: {1}, travel time: {2} mins, fleet size: {3}, wait time: {4} mins}}".format(self.bus_line,self.frequency,self.travel_time/60, self.no_of_busses, self.bus_waiting_time/60)

    def bus_count(self):
        return len(self.bus_lines)
