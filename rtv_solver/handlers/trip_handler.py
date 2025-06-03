from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.handlers.network_handler import NetworkHandler
import numpy as np
import logging
import itertools
import multiprocessing as mp
import gurobipy as gp
from gurobipy import GRB
import time

class TripHandler:
    def __init__(self,vehicles,requests,active_requests,iteration,solver_timeout,penalty,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR,REBALANCING,rtv_timeout):
        self.trips = []
        self.shared_trips_map = {}
        self.ondemand_only_trip_map = {}
        self.solver_timeout = solver_timeout
        self.rtv_timeout = rtv_timeout
        self.vehicle_to_trips_cost_map = {}
        self.trip_to_vehicle_cost_map = {}
        self.rebalancing_assignment = {}
        self.vehicle_assignment = {}
        self.request_assignment = {}
        self.starting_time = time.time()
        self.generate_ondemand_only_trips(requests,iteration)
        self.generate_trip_costs(vehicles,MAX_THREAD_CNT,0)
        self.generate_shared_trips(vehicles,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR)
        self.assign_trips_gurobi(requests,active_requests,penalty)
        if REBALANCING:
            self.get_rebalancing_trips(vehicles,requests)

    def get_new_trip_no(self):
        return len(self.trips)

    def check_rtv_timeout(self):
        time_spent = time.time()-self.starting_time
        if time_spent > self.rtv_timeout:
            raise Exception("RTV generation timedout: {0} > {1}".format(time_spent,self.rtv_timeout))

    def get_trip_cost(self,origin,destination):
        return NetworkHandler.travel_distance(origin,destination)
    
    def generate_ondemand_only_trips(self,requests,iteration):
        for request in requests:
            origin = request.origin
            destination = request.destination
            dwell_pickup, dwell_alight, latest_pick_up_time = request.dwell_pickup, request.dwell_alight, request.latest_pick_up_time
            earliest_pick_up_time = request.pick_up_time
            # earliest_pick_up_time = current_time
            # if request.pick_up_time > current_time:
            #     earliest_pick_up_time = request.pick_up_time
            trip = self.create_trip(request,request.am_capacity, request.wc_capacity,origin,destination,earliest_pick_up_time,latest_pick_up_time ,request.earliest_arrival_time,request.latest_arrival_time,dwell_pickup, dwell_alight,iteration,allow_walk=False)
            self.trips.append(trip)
            self.ondemand_only_trip_map[request.id] = trip.number

    def create_trip(self,request,am_capacity, wc_capacity,origin,destination,pick_up_time,latest_pick_up_time,earliest_arrival_time,latest_arrival_time,dwell_pickup, dwell_alight,iteration, bus_combination=None,first_last_mile_type=None,allow_walk=True):
        if allow_walk and self.can_walk(origin,destination):
            return None
        trip_no = self.get_new_trip_no()
        cost = self.get_trip_cost(origin,destination)
        return Trip(request.id,trip_no,am_capacity, wc_capacity, pick_up_time, latest_pick_up_time, earliest_arrival_time,latest_arrival_time, origin, destination,cost,dwell_pickup, dwell_alight, iteration, bus_combination=bus_combination,first_last_mile_type=first_last_mile_type)
    
    def create_trip_for_picked_requests(boarded_requests,iteration):
        trip_no = -1
        boarded_trips = []
        for request_id in boarded_requests:
            request = boarded_requests[request_id]
            boarded_trips.append(Trip(request_id,trip_no,request.am_capacity, request.wc_capacity, request.pick_up_time, request.latest_pick_up_time, request.earliest_arrival_time,request.latest_arrival_time, request.origin, request.destination,None,request.dwell_pickup, request.dwell_alight, iteration))
            trip_no-=1
        return boarded_trips

    def get_first_mile_trip(self,request,bustrip):
        origin = request.origin
        destination = bustrip.pick_up_stop_node
        return self.create_trip(request,origin,destination,request.pick_up_time, bustrip.leaving_time,bus_combination=bustrip.id,first_last_mile_type=0)

    def get_last_mile_trip(self,request,bustrip):
        destination = request.destination
        origin = bustrip.destination_stop_node
        return self.create_trip(request,origin,destination,bustrip.arrival_time, request.arrival_time,bus_combination=bustrip.id,first_last_mile_type=1)
    
    def can_walk(self,origin,destination):
        distance = NetworkHandler.travel_distance(origin,destination)
        return distance <= self.walk_distance_cutoff

    def create_trip_cost(vehicle,trip_no,trips):
        added_cost, feasibility = VehicleHandler.add_new_trips(vehicle, trips, add=False)
        if feasibility:
            return TripCost(trip_no,vehicle.id,added_cost)
        return None
        
    def process_result(trip_cost):
        if trip_cost != None:
            TripHandler.trip_costs.append(trip_cost)

    def generate_trip_costs(self,vehicles,max_num_thread,trip_start):
        if trip_start == 0:
            TripHandler.trip_costs = []

        last_trip_cost_index = len(TripHandler.trip_costs)
        
        pool = mp.Pool(max_num_thread)
        for trip in self.trips[trip_start:]:
            trips = []
            if isinstance(trip,Trip):
                trips = [trip]
            else:
                shared_trip = trip
                for sub_trip_no in shared_trip.trips:
                    sub_trip = self.trips[sub_trip_no]
                    trips.append(sub_trip)
            selected_vehicle_ids = vehicles.keys()
            if trip_start > 0:
                selected_vehicle_ids = self.common_vehicles_of_trips(trips)
            for vehicle_id in selected_vehicle_ids:
                pool.apply_async(TripHandler.create_trip_cost, args=(vehicles[vehicle_id],trip.number,trips,), callback=TripHandler.process_result)
        pool.close()
        pool.join()

        for vehicle_id in vehicles:
            if vehicle_id not in self.vehicle_to_trips_cost_map:
                self.vehicle_to_trips_cost_map[vehicle_id] = []

        for trip in self.trips[trip_start:]:
            if trip.number not in self.trip_to_vehicle_cost_map:
                self.trip_to_vehicle_cost_map[trip.number] = []

        trip_cost_index = last_trip_cost_index
        for trip_cost in TripHandler.trip_costs[last_trip_cost_index:]:
            vehicle_id = trip_cost.vehicle_id
            trip_no = trip_cost.trip_no
            self.vehicle_to_trips_cost_map[vehicle_id].append(trip_cost_index)
            trip = self.trips[trip_no]
            if isinstance(trip,Trip):
                self.trip_to_vehicle_cost_map[trip_no].append(trip_cost_index)
            else:
                self.trip_to_vehicle_cost_map[trip_no].append(trip_cost_index)
                for sub_trip_no in trip.trips:
                    self.trip_to_vehicle_cost_map[sub_trip_no].append(trip_cost_index)
            trip_cost_index+=1

    def can_share_trips(trips,trip_nos,new_trip,current_cost,current_sequence,SHAREABLE_COST_FACTOR):
        feasible, cost, sequence = VehicleHandler.can_serve_trips(trips,new_trip,current_sequence)
        if feasible and cost <= SHAREABLE_COST_FACTOR*current_cost:
            return SharedTrip(0,trip_nos,cost,sequence)
        return None
    
    def process_shared_trip_result(shared_trip):
        if shared_trip != None:
            TripHandler.shared_trips_to_create.append(shared_trip)

    def update_shared_trip_numbers(self,cardinality):
        self.selected_combinations = []
        for shared_trip in TripHandler.shared_trips_to_create:
            new_shared_trip_no = self.get_new_trip_no()
            shared_trip.number = new_shared_trip_no
            self.trips.append(shared_trip)
            self.shared_trips_map[cardinality].append(new_shared_trip_no)
            self.selected_combinations.append(shared_trip.trips)

    def common_vehicles_of_trips(self,trips):
        common_vehicles = []
        for trip in trips:
            vehicles = []
            for trip_cost_index in self.trip_to_vehicle_cost_map[trip.number]:
                trip_cost = TripHandler.trip_costs[trip_cost_index]
                vehicles.append(trip_cost.vehicle_id)
            if len(common_vehicles) == 0:
                common_vehicles = set(vehicles)
            else:
                common_vehicles = common_vehicles.union(set(vehicles))
            if len(common_vehicles) == 0:
                return common_vehicles
        return common_vehicles

    def generate_shared_trips(self,vehicles,max_cardinality,max_num_thread,SHAREABLE_COST_FACTOR):
        cardinality = 2
        self.selected_combinations = []
        while cardinality <= max_cardinality:
            self.check_rtv_timeout()
            trip_start = len(self.trips)
            TripHandler.shared_trips_to_create = []
            self.shared_trips_to_create = []
            st = time.time()
            self.shared_trips_map[cardinality] = []
            if cardinality == 2:
                no_of_trips = len(self.trips)
                pool = mp.Pool(max_num_thread)
                for trip_nos in itertools.combinations(list(range(no_of_trips)),cardinality):
                    trip1 = self.trips[trip_nos[0]]
                    trip2 = self.trips[trip_nos[1]]
                    current_cost = trip1.cost+trip2.cost
                    trips = {}
                    for trip_no in trip_nos:
                        trip = self.trips[trip_no]
                        trips[trip.id] = trip
                    if len(self.common_vehicles_of_trips(trips.values())) > 0:
                        pool.apply_async(TripHandler.can_share_trips,args=(trips,set(trip_nos),trip1,current_cost,[],SHAREABLE_COST_FACTOR,), callback=TripHandler.process_shared_trip_result)
                pool.close()
                pool.join()
            else:
                tried_combinations = []
                prev_shared_trips = self.shared_trips_map[cardinality-1]
                pool = mp.Pool(max_num_thread)
                for shared_trip1_index in range(len(prev_shared_trips)):
                    shared_trip1 = self.trips[prev_shared_trips[shared_trip1_index]]
                    for shared_trip2_index in range(shared_trip1_index+1,len(prev_shared_trips)):
                        shared_trip2 = self.trips[prev_shared_trips[shared_trip2_index]]
                        uncommon_trips = shared_trip2.trips.difference(shared_trip1.trips)
                        if len(uncommon_trips) == 1 and len(self.common_vehicles_of_trips([shared_trip1,shared_trip2])) > 0:
                            trip = self.trips[uncommon_trips.pop()]
                            current_cost = trip.cost+shared_trip1.cost
                            trip_nos = shared_trip1.trips.copy()
                            trip_nos.add(trip.number)
                            if trip_nos not in tried_combinations:
                                tried_combinations.append(trip_nos)
                                sub_combination_found = True
                                for combination in itertools.combinations(trip_nos,cardinality-1):
                                    if set(combination) not in self.selected_combinations:
                                        sub_combination_found = False
                                        break
                                if sub_combination_found:
                                    trips = {}
                                    for trip_no in trip_nos:
                                        temp_trip = self.trips[trip_no]
                                        trips[temp_trip.id] = temp_trip
                                    if len(self.common_vehicles_of_trips(trips.values())) > 0:
                                        pool.apply_async(TripHandler.can_share_trips,args=(trips,trip_nos,trip,current_cost,shared_trip1.sequence,SHAREABLE_COST_FACTOR,), callback=TripHandler.process_shared_trip_result)
                pool.close()
                pool.join()
            
            self.update_shared_trip_numbers(cardinality)
            if len(self.shared_trips_map[cardinality]) == 0:
                break
            self.generate_trip_costs(vehicles,max_num_thread,trip_start)
            logging.debug("time to generate cardinal {0} trips: {1}".format(cardinality,time.time()-st))
            logging.debug("Number of cardinal {0} trips: {1}".format(cardinality,len(self.shared_trips_map[cardinality])))
            cardinality+=1
    
    def log_with_timestamp(self,timestamp,message):
        logging.info('{0}: {1}'.format(timestamp,message))

    def get_x(self,i):
        if self.x[i] > 0.9:
            return 1
        return 0
    
    def assign_trips_gurobi(self,requests,active_requests,penalty):
        trip_count = len(TripHandler.trip_costs)
        request_count = len(requests)

        logging.debug("Started building optimization problem")
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model('RTV assignment',env=env)
            var_type = GRB.BINARY
            trip_costs = np.zeros(trip_count)
            for i in range(trip_count):
                trip_costs[i] = TripHandler.trip_costs[i].cost
            x_t = m.addVars(trip_count,lb=0,ub=1,obj=trip_costs,name="t", vtype=var_type)

            penalties = np.ones(request_count)
            request_no = 0
            for request in requests:
                penalties[request_no] = request.priority
                if request.id in active_requests:
                    penalties[request_no] = 100
                request_no+=1
            x_r = m.addVars(request_count,lb=0,ub=1,obj=penalties*penalty,name="r", vtype=var_type)

            m.addConstrs((gp.quicksum(x_t[i] for i in self.vehicle_to_trips_cost_map[vehicle_id]) <= 1 for vehicle_id in list(self.vehicle_to_trips_cost_map.keys())), "veh")

            request_no = 0
            for request in requests:
                trip_no = self.ondemand_only_trip_map[request.id]
                cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]

                m.addConstr(x_r[request_no]+gp.quicksum(x_t[i] for i in cost_map_indices) == 1,"req_{0}".format(request.id))
                
                # all the previously assigned requests should be picked up
                # if request.id in active_requests:
                #     m.addConstr(x_r[request_no] == 0,"active_req_{0}".format(request.id))
                request_no+=1

            m.setParam('TimeLimit', self.solver_timeout)
            m.optimize()

            self.trip_sizes = []
            self.unassigned_trip_count = 0
            self.taxi_only_trip_count = 0
            self.with_one_bus_trip_count = 0
            self.with_two_bus_trip_count = 0
            self.added_distance = 0

            if m.Status == GRB.OPTIMAL or m.Status == GRB.SUBOPTIMAL:
                logging.info("Total time spent on optimization: {0}".format(m.Runtime))


                for vehicle_id in self.vehicle_to_trips_cost_map:
                    for i in self.vehicle_to_trips_cost_map[vehicle_id]:
                        if x_t[i].X == 1:
                            trip_cost = TripHandler.trip_costs[i]
                            self.added_distance+=trip_cost.cost
                            trip_no = trip_cost.trip_no
                            trip = self.trips[trip_no]
                            trips = []
                            if isinstance(trip,Trip):
                                trips.append(trip)
                            else:
                                for sub_trip_no in trip.trips:
                                    trips.append(self.trips[sub_trip_no])
                            self.trip_sizes.append(len(trips))
                            self.vehicle_assignment[vehicle_id] = trips

                for request in requests:
                    found_assignment = False
                    trip_no = self.ondemand_only_trip_map[request.id]
                    cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
                    for index in cost_map_indices:
                        if x_t[index].X == 1:
                            trip_cost = TripHandler.trip_costs[index]
                            vehicle_id = trip_cost.vehicle_id
                            self.request_assignment[request.id] = vehicle_id
                            found_assignment = True
                            self.taxi_only_trip_count+=1
                            break

                    if not found_assignment:
                        self.unassigned_trip_count+=1
            else:
                self.unassigned_trip_count = request_count
                raise Exception("Gurobi solver ended with code: {0}".format(m.Status))
            logging.info('No of requests: {0}, unassigned requests: {1}, assigned requests: {2}'.format(request_count,self.unassigned_trip_count,self.taxi_only_trip_count))

    def get_rebalancing_trips(self,vehicles,requests):
        empty_vehicles = []
        for vehicle_id in vehicles:
            if vehicle_id not in self.vehicle_assignment:
                vehicle = vehicles[vehicle_id]
                if not (vehicle.rebalancing or len(vehicle.stop_sequence)>0):
                    empty_vehicles.append(vehicle_id)
                
        unassigned_requests = []
        for request in requests:
            if request.id not in self.request_assignment:
                unassigned_requests.append(request)

        number_of_vehicles = len(empty_vehicles)
        number_of_requests = len(unassigned_requests)
        max_rebalancing_count = min(number_of_vehicles,number_of_requests)

        if max_rebalancing_count>0:
            m = gp.Model('Rebalancing')
            var_type = GRB.BINARY
            rebalancing_costs = np.zeros((number_of_vehicles,number_of_requests))
            for i in range(number_of_vehicles):
                vehicle = vehicles[empty_vehicles[i]]
                for j in range(number_of_requests):
                    origin = unassigned_requests[j].origin
                    rebalancing_costs[i][j] = VehicleHandler.cost_of_rebalancing(vehicle,origin)
            y_vr = m.addVars(number_of_vehicles,number_of_requests,lb=0,ub=1,obj=rebalancing_costs,name="y_vr", vtype=var_type)

            m.addConstrs((y_vr.sum(i,'*') <= 1 for i in range(number_of_vehicles)), "veh")
            m.addConstrs((y_vr.sum('*',j) <= 1 for j in range(number_of_requests)), "req")
            m.addConstr((y_vr.sum() <= max_rebalancing_count), "total_assignment")
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                for i in range(number_of_vehicles):
                    for j in range(number_of_requests):
                        if y_vr[i,j].X == 1:
                            vehicle_id = empty_vehicles[i]
                            origin = unassigned_requests[j].origin
                            self.rebalancing_assignment[vehicle_id] = origin
                            break
