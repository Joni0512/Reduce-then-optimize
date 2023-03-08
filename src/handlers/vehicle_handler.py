import logging
import pandas as pd
from structure.vehicle import Vehicle
from structure.vehicle_stop import VehicleStop
from handlers.network_handler import NetworkHandler
from datetime import datetime
from datetime import timedelta
import pickle

START_TIME = 'start_time'
CAPACITY = 'capacity'
START_NODE = 'node'
ID = 'id'

TYPE_PICK_UP = 0
TYPE_DROP_OFF = 1

class VehicleHandler:
    MAX_CAPACITY = 0
    LARGEST_TSP = 4
    def __init__(self, filename, output_directory, starting_date, max_number_of_vehicles, max_capacity):
        self.vehicles = {}
        self.count = 0
        VehicleHandler.MAX_CAPACITY = max_capacity
        self.read_vehicles(filename, starting_date, max_number_of_vehicles)
        self.output_directory = output_directory
        logging.info('Total No of vehicles: {0}'.format(self.count))

    def save_snapshot(self):
        with open(self.output_directory+"vehicle_snapshot.p", 'wb') as snapshot_file:
            pickle.dump(self, snapshot_file)

    def load_snapshot(snapshot_directory):
        snapshot = None
        with open(snapshot_directory+"vehicle_snapshot.p", 'rb') as snapshot_file:
            snapshot = pickle.load(snapshot_file)
        return snapshot

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
    
    def get_time_delta(seconds):
        return timedelta(seconds=int(seconds))

    def simulate_vehicle(self,current_time, vehicle):
        completed_stops = []
        if current_time >= vehicle.start_time:
            if not vehicle.started:
                vehicle.started = True
            if len(vehicle.stop_sequence) == 0:
                vehicle.time_at_last = current_time
        while len(vehicle.stop_sequence)>0 and current_time >= vehicle.time_at_next:
            next_stop = vehicle.stop_sequence.pop(0)
            # logging the stop
            next_stop.stop_time = vehicle.time_at_next
            next_stop.request_id = vehicle.trips[next_stop.trip_id].request_id
            next_stop.vehicle_id = vehicle.id
            completed_stops.append(next_stop)

            vehicle.last_node = next_stop.node
            vehicle.time_at_last = vehicle.time_at_next
            if next_stop.type == TYPE_PICK_UP:
                vehicle.picked.append(next_stop.trip_id)
                vehicle.trips[next_stop.trip_id].picked = True
            else:
                del vehicle.trips[next_stop.trip_id]
            if len(vehicle.stop_sequence) > 0:
                next_stop = vehicle.stop_sequence[0]
                vehicle.time_at_next = vehicle.time_at_last + VehicleHandler.get_time_delta(NetworkHandler.travel_time(vehicle.last_node,next_stop.node))
                next_trip = vehicle.trips[next_stop.trip_id]
                if next_stop.type == TYPE_PICK_UP and vehicle.time_at_next < next_trip.pick_up_time:
                    vehicle.time_at_next = next_trip.pick_up_time
        return completed_stops

    def simulate_vehicles(self,current_time):
        completed_stops = []
        for vehicle_id in self.vehicles:
            veh_completed_stops = self.simulate_vehicle(current_time, self.vehicles[vehicle_id])
            completed_stops.extend(veh_completed_stops)
        return completed_stops

    def get_vehicle_exact_location(self,vehicle_id,current_time):
        vehicle = self.vehicles[vehicle_id]
        next_immediate_node = vehicle.last_node
        time_at_next_immediate_node = vehicle.time_at_last
        if len(vehicle.stop_sequence)>0:
            target_node = vehicle.stop_sequence[0].node
            path = NetworkHandler.get_path(next_immediate_node,target_node)
            path.pop(0)
            while time_at_next_immediate_node < current_time and len(path)>0:
                last_node = next_immediate_node
                next_immediate_node = path.pop(0)
                time_at_next_immediate_node = time_at_next_immediate_node + VehicleHandler.get_time_delta(NetworkHandler.travel_time(last_node,next_immediate_node))
        return next_immediate_node

    def get_vehicle_locations(self,current_time):
        locations = {}
        for vehicle_id in self.vehicles:
            locations[int(vehicle_id)] = self.get_vehicle_exact_location(vehicle_id,current_time)
        return locations
    
    def add_new_trips(current_time, vehicle, new_trips, add=False):
        feasible = False
        added_cost = -1
        if vehicle.started:
            next_immediate_node = vehicle.last_node
            time_at_next_immediate_node = vehicle.time_at_last
            if len(vehicle.stop_sequence)>0:
                target_node = vehicle.stop_sequence[0].node
                path = NetworkHandler.get_path(next_immediate_node,target_node)
                path.pop(0)
                while time_at_next_immediate_node < current_time and len(path)>0:
                    last_node = next_immediate_node
                    next_immediate_node = path.pop(0)
                    time_at_next_immediate_node = time_at_next_immediate_node + VehicleHandler.get_time_delta(NetworkHandler.travel_time(last_node,next_immediate_node))
        
            sequence, cost = None, None
            if len(vehicle.trips) < VehicleHandler.LARGEST_TSP:
                trips = vehicle.trips.copy()
                for trip in new_trips:
                    trips[trip.id] = trip
                trips_to_pick_up = []
                trips_to_drop_off = []
                for trip_id in trips:
                    trips_to_drop_off.append(trip_id)
                    if not trips[trip_id].picked:
                        trips_to_pick_up.append(trip_id)
                sequence, cost, feasible = VehicleHandler.get_optimal_stop_sequence(next_immediate_node,time_at_next_immediate_node,vehicle.capacity,trips,trips_to_pick_up,trips_to_drop_off,[],0)
            else:
                sequence, cost, feasible = VehicleHandler.get_heuristic_stop_sequence(next_immediate_node,time_at_next_immediate_node,vehicle.capacity,vehicle.trips.copy(),new_trips[0],vehicle.stop_sequence)
            added_cost = cost - VehicleHandler.cost_of_serving_sequence(next_immediate_node,vehicle)
            
            if feasible and add:
                vehicle.last_node = next_immediate_node
                vehicle.time_at_last = time_at_next_immediate_node
                for trip in new_trips:
                    vehicle.trips[trip.id] = trip
                vehicle.stop_sequence = sequence
                next_stop = vehicle.stop_sequence[0]
                vehicle.time_at_next = vehicle.time_at_last + VehicleHandler.get_time_delta(NetworkHandler.travel_time(vehicle.last_node,next_stop.node))
                if next_stop.type == TYPE_PICK_UP and vehicle.time_at_next < vehicle.trips[next_stop.trip_id].pick_up_time:
                    vehicle.time_at_next = vehicle.trips[next_stop.trip_id].pick_up_time
    
        return added_cost,feasible

    def cost_of_serving_sequence(next_immediate_node,vehicle):
        cost = 0
        last_node = next_immediate_node
        for stop in vehicle.stop_sequence:
            cost += NetworkHandler.travel_distance(last_node,stop.node)
            last_node = stop.node
        return cost
    
    def get_optimal_stop_sequence(last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,sequence,cost):
        if len(trips_to_pick_up) == 0 and len(trips_to_drop_off) == 0:
            return sequence, cost, True
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        if len(trips_to_drop_off) - len(trips_to_pick_up) < max_capacity:
            for trip_id in trips_to_pick_up:
                trip = trips[trip_id]
                travel_time = NetworkHandler.travel_time(last_node,trip.origin)
                time_at_pick_up = time_at_last_node + VehicleHandler.get_time_delta(travel_time)
                if time_at_pick_up < trip.pick_up_time:
                    time_at_pick_up = trip.pick_up_time
                
                new_cost = cost + NetworkHandler.travel_distance(last_node,trip.origin)
                new_trips_to_pick_up = trips_to_pick_up.copy()
                new_trips_to_pick_up.remove(trip_id)
                new_sequence = sequence.copy()
                new_sequence.append(VehicleStop(trip_id,trip.origin,TYPE_PICK_UP))
                new_sequence, new_cost, new_feasible = VehicleHandler.get_optimal_stop_sequence(trip.origin,time_at_pick_up,max_capacity,trips,new_trips_to_pick_up,trips_to_drop_off,new_sequence,new_cost)
                if new_feasible:
                    if (not feasible) or (current_lowest_cost > new_cost):
                        current_lowest_cost = new_cost
                        feasible = new_feasible
                        best_sequence = new_sequence
        
        for trip_id in trips_to_drop_off:
            if trip_id not in trips_to_pick_up:
                trip = trips[trip_id]
                travel_time = NetworkHandler.travel_time(last_node,trip.destination)
                time_at_drop_off = time_at_last_node + VehicleHandler.get_time_delta(travel_time)
                if time_at_drop_off <= trip.arrival_time:
                    new_cost = cost + NetworkHandler.travel_distance(last_node,trip.destination)
                    new_trips_to_drop_off = trips_to_drop_off.copy()
                    new_trips_to_drop_off.remove(trip_id)
                    new_sequence = sequence.copy()
                    new_sequence.append(VehicleStop(trip_id,trip.destination,TYPE_DROP_OFF))
                    new_sequence, new_cost, new_feasible = VehicleHandler.get_optimal_stop_sequence(trip.destination,time_at_drop_off,max_capacity,trips,trips_to_pick_up,new_trips_to_drop_off,new_sequence,new_cost)
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > new_cost):
                            current_lowest_cost = new_cost
                            feasible = new_feasible
                            best_sequence = new_sequence

        return best_sequence, current_lowest_cost, feasible

    def get_heuristic_stop_sequence(last_node,time_at_last_node,max_capacity,trips,new_trip,existing_sequence):
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        load = 2*len(trips) - len(existing_sequence)
        for pick_up_index in range(len(existing_sequence)+1):
            for drop_off_index in range(pick_up_index+1,len(existing_sequence)+2):
                cost = 0
                new_feasible = True
                new_sequence = existing_sequence.copy()
                trip_id = new_trip.id
                new_sequence.insert(pick_up_index,VehicleStop(trip_id,new_trip.origin,TYPE_PICK_UP))
                new_sequence.insert(drop_off_index,VehicleStop(trip_id,new_trip.destination,TYPE_DROP_OFF))
                
                current_time = time_at_last_node
                current_node = last_node
                current_load = load
                for stop in new_sequence:
                    trip = new_trip
                    if stop.trip_id != trip_id:
                        trip = trips[stop.trip_id]
                    travel_time = VehicleHandler.get_time_delta(NetworkHandler.travel_time(current_node,stop.node))
                    cost +=  NetworkHandler.travel_distance(current_node,stop.node)
                    current_time = current_time + travel_time
                    if stop.type == TYPE_PICK_UP:
                        if current_time < trip.pick_up_time:
                            current_time = trip.pick_up_time
                        current_load+=1
                        if current_load > max_capacity:
                            new_feasible = False
                            break
                    else:
                        current_load-=1
                        if current_time > trip.arrival_time:
                            new_feasible = False
                            break
                    current_node = stop.node
                if new_feasible:
                    if (not feasible) or (current_lowest_cost > cost):
                        current_lowest_cost = cost
                        feasible = new_feasible
                        best_sequence = new_sequence
                        
        return best_sequence, current_lowest_cost, feasible

    def can_serve_trips(current_time,trips):
        trips_to_pick_up = []
        trips_to_drop_off = []
        for trip_id in trips:
            trips_to_pick_up.append(trip_id)
            trips_to_drop_off.append(trip_id)
        for trip_id in trips:
            pick_up_location = trips[trip_id].origin
            _,cost,feasible = VehicleHandler.get_optimal_stop_sequence(pick_up_location,current_time,VehicleHandler.MAX_CAPACITY,trips,trips_to_pick_up,trips_to_drop_off,[],0)
            if feasible:
                return feasible,cost
        return False,None
