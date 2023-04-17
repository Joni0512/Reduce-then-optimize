import logging
from datetime import timedelta
from datetime import datetime
from structure.request import Request
from handlers.network_handler import NetworkHandler
from structure.bus_line import BusLine
from structure.bus_run import BusRun
from structure.bustrip import BusTrip
import gtfs_kit as gk
import pandas as pd
import numpy as np
import pickle

class BusHandler:
    def __init__(self, bus_directory, bus_starting_time,  capacity,load_saved_busslines):
        self.busslines = {}
        if load_saved_busslines:
            with open(bus_directory+"buslines.obj", 'rb') as filehandler:
                self.busslines = pickle.load(filehandler)
        else:
            feed = gk.read_feed(bus_directory, dist_units='km')
            stop_map = pd.read_csv(bus_directory+"stop_map_10km.csv")
            stop_map_dic = {}
            for _,row in stop_map.iterrows():
                stop_id = None
                try:
                    stop_id = int(row.stop_id)
                except ValueError:
                    stop_id = row.stop_id
                stop_map_dic[stop_id] = int(row.node_id)

            selected_day = None
            for day in feed.get_first_week():
                year = int(day[:4])
                month = int(day[4:6])
                date = int(day[6:])
                date_time_obj = datetime(year=year,month=month,day=date)
                if date_time_obj.weekday() == bus_starting_time.weekday():
                    selected_day = day
                    break
            for _,row in feed.routes.iterrows():
                route_id = row.route_id
                timetable = feed.build_route_timetable(route_id, [selected_day])
                if timetable.shape[0] > 0:
                    for direction_id in timetable.direction_id.unique():
                        dir_timetable = timetable[timetable.direction_id == direction_id]
                        unique_stops = []
                        for stop in dir_timetable.stop_id.unique():
                            try:
                                unique_stops.append(stop_map_dic[int(stop)])
                            except ValueError:
                                unique_stops.append(stop_map_dic[stop])
                        busline = BusLine(route_id,unique_stops,capacity)
                        for trip_id in dir_timetable.trip_id.unique():
                            stops = []
                            arrival_times = []
                            departure_times = []
                            for _,stop in dir_timetable[dir_timetable.trip_id == trip_id].sort_values('stop_sequence').iterrows():
                                stops.append(stop.stop_id)
                                arrival_times.append(self.str_to_datetime(bus_starting_time,stop.arrival_time))
                                departure_times.append(self.str_to_datetime(bus_starting_time,stop.departure_time))
                            real_stops = []
                            for stop in stops:
                                try:
                                    real_stops.append(stop_map_dic[int(stop)])
                                except ValueError:
                                    real_stops.append(stop_map_dic[stop])
                            bus_run = BusRun(real_stops,arrival_times,departure_times)
                            bus_run.load = np.zeros((len(real_stops)),dtype=np.int16)
                            busline.bus_runs.append(bus_run)
                        self.busslines["{0}_{1}".format(route_id,direction_id)] = busline
            with open(bus_directory+"buslines.obj", 'wb') as filehandler:
                pickle.dump(self.busslines,filehandler)
        logging.info('Total No of bus lines: {0}'.format(len(self.busslines)))
        eligible_lines = pd.read_csv(bus_directory+"eligible_lines_10km.csv")
        self.eligible_bus_lines = {}
        for i in range(1,NetworkHandler.get_network_size()+2):
            self.eligible_bus_lines[i] = {}
        for _,row in eligible_lines.iterrows():
            node = int(row.node)
            stop = int(row.stop)
            direction = row.direction
            line = row.line
            self.eligible_bus_lines[node]["{0}_{1}".format(line,direction)] = stop

        # for i in nodes:
        #     self.eligible_bus_lines[i] = {}
        #     for bus_line_name in self.busslines:
        #         bus_line = self.busslines[bus_line_name]
        #         closest_stop = bus_line.stops[0]
        #         closest_distance = cut_off+1
        #         for stop in bus_line.stops:
        #             dist_to_stop = NetworkHandler.travel_distance(i,stop)
        #             if dist_to_stop < closest_distance:
        #                 closest_distance = dist_to_stop
        #                 closest_stop = stop
                
        #         if closest_distance <= cut_off:
        #             self.eligible_bus_lines[i][bus_line] = closest_stop
        
    def str_to_datetime(self,bus_starting_time,str_obj):
        hour = int(str_obj[:2])
        minutes = int(str_obj[3:5])
        seconds = int(str_obj[6:])
        return bus_starting_time + timedelta(hours=hour, minutes=minutes,seconds=seconds)
    
    def get_time_delta(self,seconds):
        return timedelta(seconds=int(seconds))
    
    def get_departure_arrival_index(self,bus_run,source_stop, destination_stop):
        if source_stop not in bus_run.stops or destination_stop not in bus_run.stops:
            return 0,-1
        source_index = bus_run.stops.index(source_stop)
        destination_index = bus_run.stops.index(destination_stop)
        # departure stop should come before destination stop
        if source_index > destination_index:
            destination_index = len(bus_run.stops) - bus_run.stops[::-1].index(destination_stop) - 1
        return source_index,destination_index
    
    def can_add_passenger(self,bus_run, source_index,destination_index,capacity):
        # print(bus_run.stops, source_index,destination_index,capacity)
        while source_index < destination_index:
            if bus_run.load[source_index] >= capacity:
                return False
            source_index+=1
        return True
    
    def bus_trips(self,bus_line_name,source_stop, destination_stop, earliest_pick_up_time,latest_arrival_time):
        # print(bus_line_name,source_stop, destination_stop)
        bus_line = self.busslines[bus_line_name]
        run_number = bus_line.first_incomplete_run
        trips = []
        while run_number < len(bus_line.bus_runs):
            bus_run = bus_line.bus_runs[run_number]
            source_index,destination_index = self.get_departure_arrival_index(bus_run,source_stop, destination_stop)
            # departure stop should come before arrival stop
            if source_index > destination_index:
                break
            departure_time = bus_run.departure_times[source_index]
            arrival_time = bus_run.arrival_times[destination_index]
            if arrival_time > latest_arrival_time:
                break
            if self.can_add_passenger(bus_run,source_index,destination_index,bus_line.capacity):
                if departure_time >= earliest_pick_up_time:
                    bus_trip = BusTrip(bus_line_name, run_number,source_stop, destination_stop, departure_time, arrival_time)
                    trips.append(bus_trip)
            run_number+=1
        return trips

    def bus_trips_with_transfer(self,bus_line1_name, bus_line2_name,source_stop, transfer_stop, destination_stop, earliest_pick_up_time,latest_arrival_time):
        # print(bus_line1_name, bus_line2_name,source_stop, transfer_stop, destination_stop)
        bus_line1 = self.busslines[bus_line1_name]
        bus_line2 = self.busslines[bus_line2_name]

        run_number1 = bus_line1.first_incomplete_run
        trips = []
        while run_number1 < len(bus_line1.bus_runs):
            bus_run1 = bus_line1.bus_runs[run_number1]
            source_index,transfer_index_line1 = self.get_departure_arrival_index(bus_run1,source_stop, transfer_stop)
            # departure stop should come before arrival stop
            if source_index > transfer_index_line1:
                break
            departure_time_line1 = bus_run1.departure_times[source_index]
            arrival_time_line1 = bus_run1.arrival_times[transfer_index_line1]
            if arrival_time_line1 > latest_arrival_time:
                break
            if self.can_add_passenger(bus_run1,source_index,transfer_index_line1,bus_line1.capacity):
                if departure_time_line1 >= earliest_pick_up_time:
                    run_number2 = bus_line2.first_incomplete_run
                    while run_number2 < len(bus_line2.bus_runs):
                        bus_run2 = bus_line2.bus_runs[run_number2]
                        transfer_index_line2,destination_index_line2 = self.get_departure_arrival_index(bus_run2,transfer_stop, destination_stop)
                        # departure stop should come before arrival stop
                        if transfer_index_line2 > destination_index_line2:
                            break
                        departure_time_line2 = bus_run2.departure_times[transfer_index_line2]
                        arrival_time_line2 = bus_run2.arrival_times[destination_index_line2]
                        if arrival_time_line2 > latest_arrival_time:
                            break
                        if self.can_add_passenger(bus_run2,transfer_index_line2,destination_index_line2,bus_line2.capacity):
                            if departure_time_line2 >= arrival_time_line1:
                                bus_trip = BusTrip(bus_line1_name, run_number1, source_stop, destination_stop, departure_time_line1, arrival_time_line2,transfer_stop,bus_line2_name,run_number2,arrival_time_line1,departure_time_line2)
                                trips.append(bus_trip)
                        run_number2+=1
            run_number1+=1
        return trips


    def generate_bus_trips(self,request,allow_busses,allow_bus_transfers,walk_distance_cutoff):
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
            for bus_line in self.eligible_bus_lines[origin]:
                bus_line_close_to_origin.append(bus_line)
            
            bus_line_close_to_destination = []
            for bus_line in self.eligible_bus_lines[destination]:
                bus_line_close_to_destination.append(bus_line)
            
            for bus_line in bus_line_close_to_origin:
                if bus_line in bus_line_close_to_destination:
                    source_stop = self.eligible_bus_lines[origin][bus_line]
                    destination_stop = self.eligible_bus_lines[destination][bus_line]
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
                        if distance_to_source <= walk_distance_cutoff and distance_from_arrival <= walk_distance_cutoff:
                            trips = {}
                            trips[bus_line] = trips_from_line[:1]
                            return {bus_line:trips_from_line[:1]}
                        trips[bus_line] = trips_from_line[-1:]
            
            if allow_bus_transfers:
                for bus_line1 in bus_line_close_to_origin:
                    bus1 = self.busslines[bus_line1]
                    for bus_line2 in self.eligible_bus_lines[destination]:
                        if bus_line1 != bus_line2:
                            bus2 = self.busslines[bus_line2]
                            transfer_stop = -1
                            for stop in bus1.stops:
                                if stop in bus2.stops:
                                    transfer_stop = stop
                                    break
                            if transfer_stop != -1:
                                source_stop = self.eligible_bus_lines[origin][bus_line1]
                                destination_stop = self.eligible_bus_lines[destination][bus_line2]
                                first_mile_distance = NetworkHandler.travel_distance(origin, source_stop)
                                last_mile_distance = NetworkHandler.travel_distance(destination_stop, destination)
                                added_cost = first_mile_distance + last_mile_distance - trip_distance
                                if (source_stop == destination_stop or (source_stop == transfer_stop or destination_stop == transfer_stop)) or added_cost > 0:
                                    continue
                                earliest_pick_up_time = pick_up_time + self.get_time_delta(NetworkHandler.travel_time(origin,source_stop))
                                latest_arrival_time = arrival_time - self.get_time_delta(NetworkHandler.travel_time(destination_stop,destination))
                                # if first_mile_distance <= WALK_DISTANCE_CUT_OFF and last_mile_distance <= WALK_DISTANCE_CUT_OFF:
                                #     if trip_distance > 2000:
                                #         latest_arrival_time = pick_up_time+self.get_time_delta(duration*(4/3))
                                trips_from_line = self.bus_trips_with_transfer(bus_line1,bus_line2,source_stop,transfer_stop, destination_stop, earliest_pick_up_time,latest_arrival_time)
                                if len(trips_from_line) > 0:
                                    if first_mile_distance < walk_distance_cutoff and last_mile_distance < walk_distance_cutoff:
                                        trips = {}
                                        trips["{0},{1}".format(bus_line1,bus_line2)] = trips_from_line[:1]
                                        return trips
                                    trips["{0},{1}".format(bus_line1,bus_line2)] = trips_from_line[-1:]

        return trips

    def update_completed_bus_runs(self,current_time):
        for bus_line_label in self.busslines:
            bus_line = self.busslines[bus_line_label]
            run_number = bus_line.first_incomplete_run
            while run_number < len(bus_line.bus_runs):
                bus_run = bus_line.bus_runs[run_number]
                if bus_run.arrival_times[-1] < current_time:
                    break
                run_number+=1
            bus_line.first_incomplete_run = run_number

    def add_passenger_to_bus_run(self,bus_trip):
        if len(bus_trip.bus_lines) == 1:
            bus_line = self.busslines[bus_trip.bus_lines[0]]
            bus_run = bus_line.bus_runs[bus_trip.bus_run[0]]
            source_index,dest_index = self.get_departure_arrival_index(bus_run,bus_trip.pick_up_stop, bus_trip.destination_stop)
            while source_index < dest_index:
                bus_run.load[source_index] = bus_run.load[source_index] + 1
                source_index+=1
        else:
            bus_line1 = self.busslines[bus_trip.bus_lines[0]]
            bus_run1 = bus_line1.bus_runs[bus_trip.bus_run[0]]
            source_index,dest_index = self.get_departure_arrival_index(bus_run1,bus_trip.pick_up_stop, bus_trip.transfer_point)
            while source_index < dest_index:
                bus_run1.load[source_index] = bus_run1.load[source_index] + 1
                source_index+=1
            bus_line2 = self.busslines[bus_trip.bus_lines[1]]
            bus_run2 = bus_line2.bus_runs[bus_trip.bus_run[1]]
            source_index,dest_index = self.get_departure_arrival_index(bus_run2,bus_trip.transfer_point, bus_trip.destination_stop)
            while source_index < dest_index:
                bus_run2.load[source_index] = bus_run2.load[source_index] + 1
                source_index+=1
