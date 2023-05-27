import logging
import pandas as pd
from structure.vehicle import Vehicle
from structure.vehicle_stop import VehicleStop
from structure.node import Node
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
            capacity = min(int(row[CAPACITY]),self.MAX_CAPACITY)
            id = int(row[ID])
            start_time = starting_date + timedelta(hours=row[START_TIME].hour,minutes=row[START_TIME].minute,seconds=row[START_TIME].second)
            nearest_lat,nearest_lon = NetworkHandler.get_nearest_node(float(row.lat),float(row.lon))
            vehicle = Vehicle(id,Node(nearest_lat,nearest_lon) , capacity, start_time)
            self.vehicles[id] = vehicle
            self.MAX_CAPACITY = max(capacity,self.MAX_CAPACITY)
            if self.count == max_number_of_vehicles:
                break
    
    def get_time_delta(seconds):
        return timedelta(seconds=int(seconds))

    def simulate_vehicle(self,current_time, vehicle):
        completed_stops = []
        picked_requests = []
        completed_requests = []
        if current_time >= vehicle.start_time:
            if not vehicle.started:
                vehicle.started = True
            if len(vehicle.stop_sequence) == 0:
                vehicle.time_at_last = current_time
        if vehicle.rebalancing and current_time >= vehicle.time_at_next:
            next_stop = vehicle.stop_sequence.pop(0)
            vehicle.last_node = next_stop.node
            vehicle.time_at_last = current_time
            vehicle.rebalancing = False
            # logging the stop
            next_stop.stop_time = vehicle.time_at_next
            next_stop.vehicle_id = vehicle.id
            completed_stops.append(next_stop)
            return completed_stops, picked_requests, completed_requests
    
        if vehicle.dwelling and vehicle.time_at_last <= current_time:
            vehicle.dwelling = False

        while len(vehicle.stop_sequence)>0 and current_time >= vehicle.time_at_next:
            next_stop = vehicle.stop_sequence.pop(0)
            # logging the stop
            next_stop.stop_time = vehicle.time_at_next
            next_stop.request_id = vehicle.trips[next_stop.trip_id].request_id
            next_stop.vehicle_id = vehicle.id
            completed_stops.append(next_stop)

            vehicle.last_node = next_stop.node
            vehicle.time_at_last = vehicle.time_at_next + VehicleHandler.get_time_delta(next_stop.dwell)
            if vehicle.time_at_last > current_time:
                vehicle.dwelling = True
            if next_stop.type == TYPE_PICK_UP:
                vehicle.picked.append(next_stop.trip_id)
                picked_trip = vehicle.trips[next_stop.trip_id]
                picked_trip.picked = True
                picked_requests.append(picked_trip.request_id)
            else:
                vehicle.picked.remove(next_stop.trip_id)
                completed_requests.append(vehicle.trips[next_stop.trip_id].request_id)
                del vehicle.trips[next_stop.trip_id]
            if len(vehicle.stop_sequence) > 0:
                next_stop = vehicle.stop_sequence[0]
                vehicle.time_at_next = vehicle.time_at_last + VehicleHandler.get_time_delta(NetworkHandler.travel_time(vehicle.last_node,next_stop.node))
                next_trip = vehicle.trips[next_stop.trip_id]
                if next_stop.type == TYPE_PICK_UP and vehicle.time_at_next < next_trip.pick_up_time:
                    vehicle.time_at_next = next_trip.pick_up_time
        
        if len(vehicle.stop_sequence)>0:
            ori,dest = vehicle.last_node,vehicle.stop_sequence[0].node
            next_immediate_node,time_at_next_immediate_node = vehicle.last_node,vehicle.time_at_last
            if not vehicle.dwelling:
                time_at_next_immediate_node,next_immediate_node = NetworkHandler.get_current_location_time(ori,dest,vehicle.time_at_last,current_time)
            vehicle.time_at_next_immediate_node = time_at_next_immediate_node
            vehicle.next_immediate_node = next_immediate_node
            vehicle.last_node,vehicle.time_at_last = next_immediate_node, time_at_next_immediate_node
            if not vehicle.rebalancing:
                updated_trip_list = {}
                trips_to_drop_off = []
                for trip_id in vehicle.trips:
                    if trip_id in vehicle.picked:
                        updated_trip_list[trip_id] = vehicle.trips[trip_id]
                        trips_to_drop_off.append(trip_id)
                vehicle.trips = updated_trip_list
                existing_sequence = []
                nodes = [vehicle.next_immediate_node]
                for stop in vehicle.stop_sequence:
                    if stop.trip_id in updated_trip_list and stop.type == TYPE_DROP_OFF:
                        existing_sequence.append(stop)
                        nodes.append(stop.node)
                vehicle.stop_sequence = existing_sequence
                if len(vehicle.picked) > 0:
                    tt_matrix, node_indices = NetworkHandler.get_travel_time_matrix(nodes)
                    best_sequence, _, _ = VehicleHandler.get_optimal_stop_sequence(next_immediate_node,time_at_next_immediate_node,vehicle.capacity,updated_trip_list,[],trips_to_drop_off,existing_sequence,tt_matrix, node_indices)
                    vehicle.stop_sequence = best_sequence
                    next_stop = vehicle.stop_sequence[0]
                    vehicle.time_at_next = time_at_next_immediate_node + VehicleHandler.get_time_delta(NetworkHandler.travel_time(next_immediate_node,next_stop.node))
        return completed_stops, picked_requests, completed_requests

    def simulate_vehicles(self,current_time):
        completed_stops = []
        picked_requests = []
        completed_requests = []
        for vehicle_id in self.vehicles:
            veh_completed_stops, veh_picked_requests, veh_completed_requests = self.simulate_vehicle(current_time, self.vehicles[vehicle_id])
            completed_stops.extend(veh_completed_stops)
            picked_requests.extend(veh_picked_requests)
            completed_requests.extend(veh_completed_requests)
        return completed_stops, picked_requests, completed_requests

    def get_vehicle_exact_location(self,vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        next_immediate_node = vehicle.last_node
        if len(vehicle.stop_sequence)>0:
            next_immediate_node = vehicle.next_immediate_node
        return next_immediate_node

    def get_vehicle_locations(self):
        locations = {}
        for vehicle_id in self.vehicles:
            locations[int(vehicle_id)] = self.get_vehicle_exact_location(vehicle_id)
        return locations

    def add_rebalancing_trip(vehicle,destination,current_time):
        vehicle.rebalancing = True
        vehicle.time_at_last = current_time
        vehicle.stop_sequence = [VehicleStop(None,destination,2,0)]
        vehicle.time_at_next = vehicle.time_at_last + VehicleHandler.get_time_delta(NetworkHandler.travel_time(vehicle.last_node,destination))
    
    def add_new_trips(current_time, vehicle, new_trips, add=False):
        feasible = False
        added_cost = -1
        if vehicle.started:
            next_immediate_node = vehicle.last_node
            time_at_next_immediate_node = vehicle.time_at_last
            if len(vehicle.stop_sequence)>0:
                time_at_next_immediate_node = vehicle.time_at_next_immediate_node
                next_immediate_node = vehicle.next_immediate_node
        
            sequence, cost = None, None
            trips_to_pick_up = []
            trips_to_drop_off = []
            trips = vehicle.trips.copy()
            nodes = [next_immediate_node]
            for trip_id in trips:
                trips_to_drop_off.append(trip_id)
                nodes.append(trips[trip_id].destination)
            for trip in new_trips:
                trips[trip.id] = trip
                trips_to_drop_off.append(trip.id)
                trips_to_pick_up.append(trip.id)
                nodes.append(trip.origin)
                nodes.append(trip.destination)
            existing_sequence = vehicle.stop_sequence
            if vehicle.rebalancing:
                existing_sequence = []
            tt_matrix, node_indices = NetworkHandler.get_travel_time_matrix(nodes)
            sequence, cost, feasible = VehicleHandler.get_optimal_stop_sequence(next_immediate_node,time_at_next_immediate_node,vehicle.capacity,trips,trips_to_pick_up,trips_to_drop_off,existing_sequence,tt_matrix, node_indices)
            added_cost = cost - VehicleHandler.cost_of_serving_sequence(next_immediate_node,vehicle,tt_matrix, node_indices)
            
            if feasible and add:
                vehicle.rebalancing = False
                vehicle.last_node = next_immediate_node
                vehicle.time_at_last = time_at_next_immediate_node
                for trip in new_trips:
                    vehicle.trips[trip.id] = trip
                vehicle.stop_sequence = sequence
                next_stop = vehicle.stop_sequence[0]
                travel_time = NetworkHandler.travel_time_from_matrix(vehicle.last_node,next_stop.node,tt_matrix, node_indices)
                vehicle.time_at_next = vehicle.time_at_last + VehicleHandler.get_time_delta(travel_time)
                if next_stop.type == TYPE_PICK_UP and vehicle.time_at_next < vehicle.trips[next_stop.trip_id].pick_up_time:
                    vehicle.time_at_next = vehicle.trips[next_stop.trip_id].pick_up_time
    
        return added_cost,feasible

    def cost_of_serving_sequence(next_immediate_node,vehicle,tt_matrix, node_indices):
        if vehicle.rebalancing:
            return 0
        cost = 0
        last_node = next_immediate_node
        for stop in vehicle.stop_sequence:
            cost += NetworkHandler.travel_time_from_matrix(last_node,stop.node,tt_matrix, node_indices)
            last_node = stop.node
        return cost

    def cost_of_rebalancing(vehicle,destination):
        return NetworkHandler.travel_distance(vehicle.last_node,destination)
    
    def get_optimal_stop_sequence(last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,existing_sequence,tt_matrix, node_indices):
        if len(trips_to_pick_up)+len(trips_to_drop_off) <= VehicleHandler.LARGEST_TSP:
            return VehicleHandler.get_exact_stop_sequence(last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,[],0,tt_matrix, node_indices)
        else:
            return VehicleHandler.get_heuristic_stop_sequence(last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,existing_sequence,tt_matrix, node_indices)
    
    def get_exact_stop_sequence(last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,sequence,cost,tt_matrix, node_indices):
        if len(trips_to_pick_up) == 0 and len(trips_to_drop_off) == 0:
            return sequence, cost, True
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        if len(trips_to_drop_off) - len(trips_to_pick_up) < max_capacity:
            for trip_id in trips_to_pick_up:
                trip = trips[trip_id]
                travel_time = NetworkHandler.travel_time_from_matrix(last_node,trip.origin,tt_matrix, node_indices)
                time_at_pick_up = time_at_last_node + VehicleHandler.get_time_delta(travel_time)
                if time_at_pick_up < trip.pick_up_time:
                    time_at_pick_up = trip.pick_up_time

                if time_at_pick_up <= trip.latest_pick_up_time:
                    time_at_pick_up = time_at_pick_up + VehicleHandler.get_time_delta(trip.dwell_pickup)
                
                    new_cost = cost + NetworkHandler.travel_distance(last_node,trip.origin)
                    new_trips_to_pick_up = trips_to_pick_up.copy()
                    new_trips_to_pick_up.remove(trip_id)
                    new_sequence = sequence.copy()
                    new_sequence.append(VehicleStop(trip_id,trip.origin,TYPE_PICK_UP,trip.dwell_pickup))
                    new_sequence, new_cost, new_feasible = VehicleHandler.get_exact_stop_sequence(trip.origin,time_at_pick_up,max_capacity,trips,new_trips_to_pick_up,trips_to_drop_off,new_sequence,new_cost,tt_matrix, node_indices)
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > new_cost):
                            current_lowest_cost = new_cost
                            feasible = new_feasible
                            best_sequence = new_sequence
        
        for trip_id in trips_to_drop_off:
            if trip_id not in trips_to_pick_up:
                trip = trips[trip_id]
                travel_time = NetworkHandler.travel_time_from_matrix(last_node,trip.destination,tt_matrix, node_indices)
                time_at_drop_off = time_at_last_node + VehicleHandler.get_time_delta(travel_time)
                if time_at_drop_off <= trip.arrival_time:
                    time_at_drop_off = time_at_drop_off + VehicleHandler.get_time_delta(trip.dwell_alight)
                    new_cost = cost + NetworkHandler.travel_distance(last_node,trip.destination)
                    new_trips_to_drop_off = trips_to_drop_off.copy()
                    new_trips_to_drop_off.remove(trip_id)
                    new_sequence = sequence.copy()
                    new_sequence.append(VehicleStop(trip_id,trip.destination,TYPE_DROP_OFF,trip.dwell_alight))
                    new_sequence, new_cost, new_feasible = VehicleHandler.get_exact_stop_sequence(trip.destination,time_at_drop_off,max_capacity,trips,trips_to_pick_up,new_trips_to_drop_off,new_sequence,new_cost,tt_matrix, node_indices)
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > new_cost):
                            current_lowest_cost = new_cost
                            feasible = new_feasible
                            best_sequence = new_sequence

        return best_sequence, current_lowest_cost, feasible

    def get_heuristic_stop_sequence(last_node,time_at_last_node,max_capacity,trips,trips_to_pick_up,trips_to_drop_off,existing_sequence,tt_matrix, node_indices):
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        load = len(trips_to_drop_off) - len(trips_to_pick_up)
        for trip_id in trips_to_pick_up:
            new_trip = trips[trip_id]
            feasible = False
            best_sequence = None
            current_lowest_cost = -1
            for pick_up_index in range(len(existing_sequence)+1):
                for drop_off_index in range(pick_up_index+1,len(existing_sequence)+2):
                    cost = 0
                    new_feasible = True
                    new_sequence = existing_sequence.copy()
                    trip_id = new_trip.id
                    new_sequence.insert(pick_up_index,VehicleStop(trip_id,new_trip.origin,TYPE_PICK_UP,new_trip.dwell_pickup))
                    new_sequence.insert(drop_off_index,VehicleStop(trip_id,new_trip.destination,TYPE_DROP_OFF,new_trip.dwell_alight))
                    
                    current_time = time_at_last_node
                    current_node = last_node
                    current_load = load
                    for stop in new_sequence:
                        trip = new_trip
                        if stop.trip_id != trip_id:
                            trip = trips[stop.trip_id]
                        travel_time = VehicleHandler.get_time_delta(NetworkHandler.travel_time_from_matrix(current_node,stop.node,tt_matrix, node_indices))
                        cost +=  NetworkHandler.travel_distance(current_node,stop.node)
                        current_time = current_time + travel_time
                        if stop.type == TYPE_PICK_UP:
                            if current_time < trip.pick_up_time:
                                current_time = trip.pick_up_time
                            current_load+=1
                            if current_load > max_capacity or current_time > trip.latest_pick_up_time:
                                new_feasible = False
                                break
                            current_time = current_time + VehicleHandler.get_time_delta(trip.dwell_pickup)
                        else:
                            current_load-=1
                            if current_time > trip.arrival_time:
                                new_feasible = False
                                break
                            current_time = current_time + VehicleHandler.get_time_delta(trip.dwell_alight)
                        current_node = stop.node
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > cost):
                            current_lowest_cost = cost
                            feasible = new_feasible
                            best_sequence = new_sequence
            if feasible:
                existing_sequence = best_sequence
            else:
                break
        return best_sequence, current_lowest_cost, feasible

    def can_serve_trips(current_time,trips,new_trip,current_sequence):
        trips_to_pick_up = []
        trips_to_drop_off = []
        nodes = []
        for trip_id in trips:
            trips_to_pick_up.append(trip_id)
            trips_to_drop_off.append(trip_id)
            nodes.append(trips[trip_id].origin)
            nodes.append(trips[trip_id].destination)
        tt_matrix, node_indices = NetworkHandler.get_travel_time_matrix(nodes)
        best_cost = None
        feasible = False
        best_sequence = None
        starting_locations = []
        if len(current_sequence) == 0:
            for trip_id in trips:
                starting_locations.append(trips[trip_id].origin)
        else:
            starting_locations.append(current_sequence[0].node)
            starting_locations.append(trips[new_trip].origin)
        for starting_location in starting_locations:
            sequence,cost,t_feasible = None,None,None
            if 2*len(trips) <= VehicleHandler.LARGEST_TSP:
                sequence,cost,t_feasible = VehicleHandler.get_exact_stop_sequence(starting_location,current_time,VehicleHandler.MAX_CAPACITY,trips,trips_to_pick_up,trips_to_drop_off,current_sequence,0,tt_matrix, node_indices)
            else:
                sequence,cost,t_feasible = VehicleHandler.get_heuristic_stop_sequence(starting_location,current_time,VehicleHandler.MAX_CAPACITY,trips,[new_trip],[new_trip],current_sequence,tt_matrix, node_indices)
            
            if t_feasible:
                if not feasible or best_cost > cost:
                    feasible = t_feasible
                    best_cost = cost
                    best_sequence = sequence
        return feasible,best_cost,best_sequence
