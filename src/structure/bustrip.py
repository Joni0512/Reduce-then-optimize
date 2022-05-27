class BusTrip:
    def __init__(self, bus_line, pick_up_stop, destination_stop, leaving_time, arrival_time, transfer_point = -1, bus_line2= None):
        self.bus_lines = [bus_line]
        self.pick_up_stop = pick_up_stop
        self.destination_stop = destination_stop
        self.leaving_time = leaving_time
        self.arrival_time = arrival_time
        if bus_line2 != None:
            self.bus_lines.append(bus_line2)
        self.transfer_point = transfer_point
        self.first_mile_trip = 0
        self.first_mile_trip_empty = False
        self.last_mile_trip = 0
        self.last_mile_trip_empty = False
        self.id = ",".join(self.bus_lines)

    # def __str__(self):
    #     return "{{Bus line: {0}, frequency: {1}, travel time: {2} mins, fleet size: {3}, wait time: {4} mins}}".format(self.bus_line,self.frequency,self.travel_time/60, self.no_of_busses, self.bus_waiting_time/60)
