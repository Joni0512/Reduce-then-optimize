import logging
import pandas as pd
from structure.vehicle import Vehicle
from structure.vehicle_stop import VehicleStop
from datetime import datetime
from datetime import timedelta

START_TIME = 'start_time'
CAPACITY = 'capacity'
START_NODE = 'node'
ID = 'id'

TYPE_PICK_UP = 0
TYPE_DROP_OFF = 1

class VehicleHandler:
    def __init__(self, filename, starting_date, speed, max_number_of_vehicles, max_capacity):
        self.vehicles = {}
        self.count = 0
        self.MAX_CAPACITY = max_capacity
        self.read_vehicles(filename, starting_date, max_number_of_vehicles)
        self.speed = speed
        logging.info('Total No of vehicles: {0}'.format(self.count))

    def read_vehicles(self, filename,starting_date, max_number_of_vehicles):
        dateparse = lambda x: datetime.strptime(x, '%H:%M:%S')
        data = pd.read_csv(filename,parse_dates=[START_TIME],date_parser=dateparse).sort_values(by = [START_TIME])
        for _, row in data.iterrows():
            self.count+=1
            node = int(row[START_NODE])
            capacity = min(int(row[CAPACITY]),self.MAX_CAPACITY)
            id = int(row[ID])
            start_time = starting_date + timedelta(hours=row[START_TIME].hour,minutes=row[START_TIME].minute,seconds=row[START_TIME].second)
            vehicle = Vehicle(id, node, capacity, start_time)
            self.vehicles[id] = vehicle
            self.MAX_CAPACITY = max(capacity,self.MAX_CAPACITY)
            if self.count == max_number_of_vehicles:
                break
    
    def get_time_delta(self,seconds):
        return timedelta(seconds=int(seconds))

    def simulate_vehicle(self,network_handler,current_time, vehicle):
        if current_time >= vehicle.start_time:
            if not vehicle.started:
                vehicle.started = True
        while len(vehicle.stop_sequence)>0 and current_time >= vehicle.time_at_next:
            next_stop = vehicle.stop_sequence.pop(0)
            vehicle.last_node = next_stop.node
            vehicle.time_at_last = vehicle.time_at_next
            if next_stop.type == TYPE_PICK_UP:
                vehicle.picked.append(next_stop.trip_id)
                vehicle.trips[next_stop.trip_id].picked = True
            else:
                del vehicle.trips[next_stop.trip_id]
            if len(vehicle.stop_sequence) > 0:
                next_stop = vehicle.stop_sequence[0]
                vehicle.time_at_next = vehicle.time_at_last + self.get_time_delta(network_handler.travel_time(vehicle.last_node,next_stop.node))
                next_trip = vehicle.trips[next_stop.trip_id]
                if next_stop.type == TYPE_PICK_UP and vehicle.time_at_next < next_trip.pick_up_time:
                    vehicle.time_at_next = next_trip.pick_up_time
    
    def add_new_trips(self,network_handler,current_time, vehicle, new_trips, add=False):
        next_immediate_node = vehicle.last_node
        time_at_next_immediate_node = vehicle.time_at_last
        if vehicle.started:
            if len(vehicle.stop_sequence)>0:
                while time_at_next_immediate_node < current_time:
                    last_node = next_immediate_node
                    next_immediate_node = network_handler.predecessor(last_node, vehicle.stop_sequence[0])
                    time_at_next_immediate_node = time_at_next_immediate_node + self.get_time_delta(network_handler.travel_time(last_node,next_immediate_node))
            else:
                time_at_next_immediate_node = current_time
        else:
            time_at_next_immediate_node = vehicle.start_time
        
        trips = vehicle.trips.copy()
        for trip in new_trips:
            trips[trip.number] = trip
        trips_to_pick_up = []
        trips_to_drop_off = []
        for trip_id in trips:
            trips_to_drop_off.append(trip_id)
            if not trips[trip_id].picked:
                trips_to_pick_up.append(trip_id)
        sequence, cost, feasible = self.get_optimal_stop_sequence(network_handler,next_immediate_node,time_at_next_immediate_node,vehicle.capacity,trips,trips_to_pick_up,trips_to_drop_off,[],0)
        added_cost = cost - self.cost_of_serving_sequence(network_handler,next_immediate_node,vehicle)
        
        if feasible and add:
            vehicle.last_node = next_immediate_node
            vehicle.time_at_last = time_at_next_immediate_node
            for trip in new_trips:
                vehicle.trips[trip.number] = trip
            vehicle.sequence = sequence
            next_stop = vehicle.sequence[0]
            vehicle.time_at_next = vehicle.time_at_last + self.get_time_delta(network_handler.travel_time(vehicle.last_node,next_stop.node))
            if next_stop.type == TYPE_DROP_OFF and vehicle.time_at_next < vehicle.trips[next_stop.trip_id].pick_up_time:
                vehicle.time_at_next = vehicle.trips[next_stop.trip_id].pick_up_time
    
        return added_cost,feasible

    def cost_of_serving_sequence(self,network_handler,next_immediate_node,vehicle):
        cost = 0
        last_node = next_immediate_node
        for stop in vehicle.stop_sequence:
            cost += network_handler.travel_distance(last_node,stop.node)
            last_node = stop.node
        return cost
    
    def get_optimal_stop_sequence(self,network_handler,last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,sequence,cost):
        if len(trips_to_pick_up) == 0 and len(trips_to_drop_off) == 0:
            return sequence, cost, True
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        if len(trips_to_drop_off) - len(trips_to_pick_up) < max_capacity:
            for trip_id in trips_to_pick_up:
                trip = trips[trip_id]
                travel_time = network_handler.travel_time(last_node,trip.origin)
                time_at_pick_up = time_at_last_node + self.get_time_delta(travel_time)
                if time_at_pick_up < trip.pick_up_time:
                    time_at_pick_up = trip.pick_up_time
                
                new_cost = cost + network_handler.travel_distance(last_node,trip.origin)
                new_trips_to_pick_up = trips_to_pick_up.copy()
                new_trips_to_pick_up.remove(trip_id)
                new_sequence = sequence.copy()
                new_sequence.append(VehicleStop(trip_id,trip.origin,TYPE_PICK_UP))
                new_sequence, new_cost, new_feasible = self.get_optimal_stop_sequence(network_handler,trip.origin,time_at_pick_up,max_capacity,trips,new_trips_to_pick_up,trips_to_drop_off,new_sequence,new_cost)
                if new_feasible:
                    if (not feasible) or (current_lowest_cost > new_cost):
                        current_lowest_cost = new_cost
                        feasible = new_feasible
                        best_sequence = new_sequence
        
        for trip_id in trips_to_drop_off:
            if trip_id not in trips_to_pick_up:
                trip = trips[trip_id]
                travel_time = network_handler.travel_time(last_node,trip.destination)
                time_at_drop_off = time_at_last_node + self.get_time_delta(travel_time)
                if time_at_drop_off <= trip.arrival_time:
                    new_cost = cost + network_handler.travel_distance(last_node,trip.destination)
                    new_trips_to_drop_off = trips_to_drop_off.copy()
                    new_trips_to_drop_off.remove(trip_id)
                    new_sequence = sequence.copy()
                    new_sequence.append(VehicleStop(trip_id,trip.destination,TYPE_DROP_OFF))
                    new_sequence, new_cost, new_feasible = self.get_optimal_stop_sequence(network_handler,trip.destination,time_at_drop_off,max_capacity,trips,trips_to_pick_up,new_trips_to_drop_off,new_sequence,new_cost)
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > new_cost):
                            current_lowest_cost = new_cost
                            feasible = new_feasible
                            best_sequence = new_sequence

        return best_sequence, current_lowest_cost, feasible

    def can_serve_trips(self,network_handler,current_time,trips):
        trips_to_pick_up = []
        trips_to_drop_off = []
        for trip_id in trips:
            trips_to_pick_up.append(trip_id)
            trips_to_drop_off.append(trip_id)
        for trip_id in trips:
            pick_up_location = trips[trip_id].origin
            _,cost,feasible = self.get_optimal_stop_sequence(network_handler,pick_up_location,current_time,self.MAX_CAPACITY,trips,trips_to_pick_up,trips_to_drop_off,[],0)
            if feasible:
                return feasible,cost
        return False,None
