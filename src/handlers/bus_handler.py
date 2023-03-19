import logging
from datetime import timedelta
from structure.bus_line import BusLine
from structure.request import Request
from structure.bus import Bus
from structure.bustrip import BusTrip
from structure.bus_run import BusRun
from handlers.network_handler import NetworkHandler
import os
import math

class BusHandler:
    def __init__(self, bus_directory, bus_starting_time, bus_dwell,  cut_off):
        self.busslines = {}
        self.bus_dwell = bus_dwell
        self.bus_starting_time = bus_starting_time
        for bus_file in os.listdir(bus_directory):
            metadata = bus_file.split(".")[0]
            bus_line = metadata.split("_")[0]
            frequency = int(metadata.split("_")[1])
            stops = []
            with open(bus_directory+bus_file, 'r') as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    stops.append(int(line))

            travel_time = 0
            no_stops = len(stops)
            for i in range(1,no_stops):
                travel_time+= NetworkHandler.travel_time(stops[i-1],stops[i])+bus_dwell
            
            no_of_busses =  math.ceil((travel_time/(3600))*frequency)
            waiting_time = 3600*(no_of_busses/frequency)-travel_time
                
            self.busslines[bus_line] = BusLine(bus_line,frequency,stops, travel_time, no_of_busses, waiting_time)
        
        bus_fleet = 0
        for bus in self.busslines:
            bus_fleet+=self.busslines[bus].no_of_busses

        # bus_close_time_cut_off = cut_off/average_edge_speed
        self.eligible_bus_lines = []
        for i in range(1,NetworkHandler.get_network_size()+1):
            self.eligible_bus_lines.append({})
            for bus_line in self.busslines:
                bus = self.busslines[bus_line]
                closest_stop = bus.stops[0]
                closest_distance = cut_off+1
                for stop in bus.stops:
                    dist_to_stop = NetworkHandler.travel_distance(i,stop)
                    if dist_to_stop < closest_distance:
                        closest_distance = dist_to_stop
                        closest_stop = stop
                
                if closest_distance <= cut_off:
                    self.eligible_bus_lines[i-1][bus_line] = closest_stop
        
        logging.info('Total No of bus lines: {0}, fleet size: {1}'.format(len(self.busslines),bus_fleet))

    def get_time_delta(self,seconds):
        return timedelta(seconds=int(seconds))
    
    def travel_time(self,bus_line,source,destination):
        if source == destination:
            return 0

        bus = self.busslines[bus_line]
        source_index = -1
        destination_index = -1
        i = 0
        while source_index==-1 or destination_index==-1:
            if bus.stops[i] == source:
                source_index = i
            elif bus.stops[i] == destination:
                destination_index = i
            i+=1
        if destination_index == 0:
            destination_index = len(bus.stops)-1

        travel_time = self.travel_time_by_index(bus_line,source_index,destination_index)
        return travel_time,source_index,destination_index

    def travel_time_by_index(self,bus_line,source_index,destination_index):
        if source_index == destination_index:
            return 0

        travel_time = 0
        bus = self.busslines[bus_line]
        i = source_index
        while i!= destination_index:
            if i == len(bus.stops)-1:
                i = 0
                travel_time+= bus.bus_waiting_time
            travel_time+=NetworkHandler.travel_time(bus.stops[i],bus.stops[i+1])+self.bus_dwell
            i+=1

        return travel_time

    def first_bus_leave_time(self,bus_line,source_index,earliest_start_time):
        bus = self.busslines[bus_line]
        travel_time = self.travel_time_by_index( bus_line, 0, source_index)
        seconds_to_first_bus_in_hour = travel_time%3600
        seconds_to_start_time_in_hour = earliest_start_time.minute*60+earliest_start_time.second
        time_gap = 3600/bus.frequency
        busses_in_between = abs(seconds_to_first_bus_in_hour-seconds_to_start_time_in_hour) // time_gap
        if seconds_to_first_bus_in_hour >= seconds_to_start_time_in_hour:
            seconds_to_first_bus_in_hour = seconds_to_first_bus_in_hour-busses_in_between*time_gap
        else:
            seconds_to_first_bus_in_hour = seconds_to_first_bus_in_hour+(busses_in_between+1)*time_gap
        start_of_the_hour = earliest_start_time.replace(minute=0,second=0)
        return start_of_the_hour + self.get_time_delta(seconds_to_first_bus_in_hour)
    
    def bus_trips(self,bus_line,source_stop, destination_stop, ealiest_pick_up_time,latest_arrival_time):
        travel_time,source_index,_ = self.travel_time( bus_line, source_stop, destination_stop)
        first_bus_leave_time = self.first_bus_leave_time(bus_line,source_index,ealiest_pick_up_time)
        trips = []
        bus_arrival_time = first_bus_leave_time + self.get_time_delta(travel_time)
        # print(bus_line,source_stop,destination_stop,travel_time,first_bus_leave_time)
        while bus_arrival_time <= latest_arrival_time:
            bus_trip = BusTrip(bus_line, source_stop, destination_stop, first_bus_leave_time, bus_arrival_time)
            trips.append(bus_trip)
            first_bus_leave_time = first_bus_leave_time + self.get_time_delta(3600/self.busslines[bus_line].frequency)
            bus_arrival_time = first_bus_leave_time + self.get_time_delta(travel_time)
        return trips

    def bus_trips_with_transfer(self,bus_line1, bus_line2,source_stop, transfer_stop, destination_stop, ealiest_pick_up_time,latest_arrival_time):
        travel_time1,source_index,_ = self.travel_time( bus_line1, source_stop, transfer_stop)
        travel_time2,transfer_index,_ = self.travel_time( bus_line2, transfer_stop, destination_stop)
        latest_arrival_time_at_transfer = latest_arrival_time - self.get_time_delta(travel_time2)
        bus_leave_time_line1 = self.first_bus_leave_time(bus_line1,source_index,ealiest_pick_up_time)
        trips = []
        bus_arrival_time_at_transfer = bus_leave_time_line1 + self.get_time_delta(travel_time1)
        while bus_arrival_time_at_transfer <= latest_arrival_time_at_transfer:
            bus_leave_time_line2 = self.first_bus_leave_time(bus_line2,transfer_index,bus_arrival_time_at_transfer)
            destination_stop_arrival_time = bus_leave_time_line2 + self.get_time_delta(travel_time2)
            while destination_stop_arrival_time <= latest_arrival_time:
                bus_trip = BusTrip(bus_line1, source_stop, destination_stop, bus_leave_time_line1, destination_stop_arrival_time,transfer_stop,bus_line2,bus_arrival_time_at_transfer,bus_leave_time_line2)
                trips.append(bus_trip)
                bus_leave_time_line2 = bus_leave_time_line2 + self.get_time_delta(3600/self.busslines[bus_line2].frequency)
                destination_stop_arrival_time = bus_leave_time_line2 + self.get_time_delta(travel_time2)
            bus_leave_time_line1 = bus_leave_time_line1 + self.get_time_delta(3600/self.busslines[bus_line1].frequency)
            bus_arrival_time_at_transfer = bus_leave_time_line1 + self.get_time_delta(travel_time1)
        return trips

    def generate_bus_trips(self,request,allow_busses,allow_bus_transfers):
        trips = {}
        if allow_busses:
            origin = request.origin
            destination = request.destination
            pick_up_time = request.pick_up_time
            arrival_time = request.arrival_time
            # duration = (arrival_time-pick_up_time).seconds
            trip_distance = NetworkHandler.travel_distance(origin, destination)
            if origin == destination:
                return trips

            bus_line_close_to_origin = []
            for bus_line in self.eligible_bus_lines[origin-1]:
                bus_line_close_to_origin.append(bus_line)
            
            bus_line_close_to_destination = []
            for bus_line in self.eligible_bus_lines[destination-1]:
                bus_line_close_to_destination.append(bus_line)
            
            for bus_line in bus_line_close_to_origin:
                if bus_line in bus_line_close_to_destination:
                    source_stop = self.eligible_bus_lines[origin-1][bus_line]
                    destination_stop = self.eligible_bus_lines[destination-1][bus_line]
                    distance_to_source = NetworkHandler.travel_distance(origin, source_stop)
                    distance_from_arrival = NetworkHandler.travel_distance(destination_stop, destination)
                    added_cost = distance_to_source+distance_from_arrival-trip_distance
                    if source_stop == destination_stop or added_cost > 0:
                        continue
                    earliest_pick_up_time = pick_up_time + self.get_time_delta(NetworkHandler.travel_time(origin,source_stop))
                    latest_arrival_time = arrival_time - self.get_time_delta(NetworkHandler.travel_time(destination_stop,destination))
                    trips_from_line = self.bus_trips(bus_line,source_stop, destination_stop, earliest_pick_up_time,latest_arrival_time)
                    if len(trips_from_line) > 0:
                        # if distance_to_source <= self.walk_distance_cut_off:
                        #     trips[bus_line] = trips_from_line[-1:]
                        # elif distance_from_arrival <= self.walk_distance_cut_off:
                        #     trips[bus_line] = trips_from_line[-1:]
                        # else:
                        trips[bus_line] = trips_from_line[-1:]
            
            if allow_bus_transfers:
                for bus_line1 in bus_line_close_to_origin:
                    bus1 = self.busslines[bus_line1]
                    for bus_line2 in self.eligible_bus_lines[destination-1]:
                        if bus_line1 != bus_line2:
                            bus2 = self.busslines[bus_line2]
                            transfer_stop = -1
                            for stop in bus1.stops:
                                if stop in bus2.stops:
                                    transfer_stop = stop
                                    break
                            if transfer_stop != -1:
                                source_stop = self.eligible_bus_lines[origin-1][bus_line1]
                                destination_stop = self.eligible_bus_lines[destination-1][bus_line2]
                                # first_mile_distance = NetworkHandler.travel_distance(origin, source_stop)
                                # last_mile_distance = NetworkHandler.travel_distance(destination_stop, destination)
                                # added_cost = first_mile_distance + last_mile_distance - trip_distance
                                added_cost = NetworkHandler.travel_distance(origin, source_stop)+NetworkHandler.travel_distance(destination_stop, destination)-trip_distance
                                if (source_stop == destination_stop or (source_stop == transfer_stop or destination_stop == transfer_stop)) or added_cost > 0:
                                    continue
                                earliest_pick_up_time = pick_up_time + self.get_time_delta(NetworkHandler.travel_time(origin,source_stop))
                                latest_arrival_time = arrival_time - self.get_time_delta(NetworkHandler.travel_time(destination_stop,destination))
                                # if first_mile_distance <= WALK_DISTANCE_CUT_OFF and last_mile_distance <= WALK_DISTANCE_CUT_OFF:
                                #     if trip_distance > 2000:
                                #         latest_arrival_time = pick_up_time+self.get_time_delta(duration*(4/3))
                                trips_from_line = self.bus_trips_with_transfer(bus_line1,bus_line2,source_stop,transfer_stop, destination_stop, earliest_pick_up_time,latest_arrival_time)
                                if len(trips_from_line) > 0:
                                    trips["{0},{1}".format(bus_line1,bus_line2)] = trips_from_line[-1:]

        return trips

    # def get_bus_trips(request_no):
    #     return bus_handler.generate_bus_trips( batch[request_no], allow_bus, allow_bus_transfer)
    
    # def generate_bus_trip_combinations(self,requests,no_of_processes):
    #     request_bus_combinations = {}
    #     pool = multiprocessing.Pool(processes=no_of_processes)
    #     request_numbers = np.arange(len(requests))
    #     combinations = pool.map(get_bus_trips,request_numbers)
    #     request_no = 0
    #     for request in requests:
    #         request_bus_combinations[request.id] = combinations[request_no]
    #         request_no+=1
    #     return request_bus_combinations
