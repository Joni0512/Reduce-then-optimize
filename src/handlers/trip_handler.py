from structure.trip import Trip
from structure.shared_trip import SharedTrip
from structure.assignment import AssignmentWithBus
from structure.assignment import TaxiOnlyAssignment
from structure.trip_cost import TripCost
from handlers.vehicle_handler import VehicleHandler
from handlers.network_handler import NetworkHandler
import numpy as np
import mosek
import logging
import itertools
import multiprocessing as mp
from datetime import timedelta
import gurobipy as gp
from gurobipy import GRB
import time
from multiprocessing.pool import ThreadPool

class TripHandler:
    def __init__(self,current_time,vehicles,requests,request_bus_combinations,active_requests,iteration,distance_cutoff,ipm_solver_timeout,penalty,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR):
        self.trips = []
        self.shared_trips_map = {}
        self.empty_trip_count = 0
        self.ondemand_only_trip_map = {}
        self.ipm_solver_timeout = ipm_solver_timeout
        self.walk_distance_cutoff = distance_cutoff
        self.vehicle_to_trips_cost_map = {}
        self.trip_to_vehicle_cost_map = {}
        self.rebalancing_assignment = {}
        self.generate_ondemand_only_trips(requests,current_time,iteration)
        self.generate_trips_with_bus(requests,request_bus_combinations)
        logging.debug("Generated first and last mile trips. Total trips {0}".format(len(self.trips)))
        st = time.time()
        self.generate_shared_trips(current_time,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR)
        print("time to generate shared trips: ",st-time.time())
        st = time.time()
        self.generate_trip_costs(vehicles,current_time,MAX_THREAD_CNT)
        print("time to generate trip cost: ",st-time.time())
        logging.debug("Generated trip costs. Total combinations {0}".format(len(TripHandler.trip_costs)))
        self.assign_trips_gurobi(vehicles,requests,request_bus_combinations,active_requests,penalty,current_time)
        self.get_rebalancing_trips(vehicles,requests)

    def get_new_trip_no(self):
        return len(self.trips)

    def get_trip_cost(self,origin,destination):
        return NetworkHandler.travel_distance(origin,destination)
    
    def generate_ondemand_only_trips(self,requests,current_time,iteration):
        for request in requests:
            origin = request.origin
            destination = request.destination
            dwell_pickup, dwell_alight, latest_pick_up_time = request.dwell_pickup, request.dwell_alight, request.latest_pick_up_time
            trip = self.create_trip(request,origin,destination,current_time,latest_pick_up_time ,request.arrival_time,dwell_pickup, dwell_alight,iteration,allow_walk=False)
            self.trips.append(trip)
            self.ondemand_only_trip_map[request.id] = trip.number

    def generate_trips_with_bus(self,requests,request_bus_combinations):
        self.bus_combinations = 0
        for request in requests:
            if request.id in request_bus_combinations:
                for bus_combination in request_bus_combinations[request.id]:
                    combination = request_bus_combinations[request.id][bus_combination]
                    for bus_trip in combination:
                        self.bus_combinations+=1
                        first_mile_trip = self.get_first_mile_trip(request,bus_trip)
                        if first_mile_trip == None:
                            bus_trip.first_mile_trip = self.empty_trip_count
                            self.empty_trip_count+=1
                            bus_trip.first_mile_trip_empty = True
                        else:
                            bus_trip.first_mile_trip = first_mile_trip.number
                            self.trips.append(first_mile_trip)

                        last_mile_trip = self.get_last_mile_trip(request,bus_trip)
                        if last_mile_trip == None:
                            bus_trip.last_mile_trip = self.empty_trip_count
                            self.empty_trip_count+=1
                            bus_trip.last_mile_trip_empty = True
                        else:
                            bus_trip.last_mile_trip = last_mile_trip.number
                            self.trips.append(last_mile_trip)

    def create_trip(self,request,origin,destination,pick_up_time,latest_pick_up_time,arrival_time,dwell_pickup, dwell_alight,iteration, bus_combination=None,first_last_mile_type=None,allow_walk=True):
        if allow_walk and self.can_walk(origin,destination):
            return None
        trip_no = self.get_new_trip_no()
        cost = self.get_trip_cost(origin,destination)
        return Trip(request.id,trip_no,pick_up_time, latest_pick_up_time, arrival_time, origin, destination,cost,dwell_pickup, dwell_alight, iteration, bus_combination=bus_combination,first_last_mile_type=first_last_mile_type)
    
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

    def create_trip_cost(vehicle,current_time,trip_no,trips):
        added_cost, feasibility = VehicleHandler.add_new_trips(current_time, vehicle, trips, add=False)
        if feasibility:
            return TripCost(trip_no,vehicle.id,added_cost)
        return None
        
    def process_result(trip_cost):
        if trip_cost != None:
            TripHandler.trip_costs.append(trip_cost)

    def generate_trip_costs(self,vehicles,current_time,max_num_thread):
        TripHandler.trip_costs = []
        pool = mp.Pool(max_num_thread)
        for vehicle_id in vehicles:
            # inputs = []
            for trip in self.trips:
                trips = []
                if isinstance(trip,Trip):
                    trips = [trip]
                else:
                    shared_trip = trip
                    for sub_trip_no in shared_trip.trips:
                        trips.append(self.trips[sub_trip_no])
                pool.apply_async(TripHandler.create_trip_cost, args=(vehicles[vehicle_id],current_time,trip.number,trips,), callback=TripHandler.process_result)
        pool.close()
        pool.join()

        for vehicle_id in vehicles:
            self.vehicle_to_trips_cost_map[vehicle_id] = []

        for trip in self.trips:
            self.trip_to_vehicle_cost_map[trip.number] = []

        trip_cost_index = 0
        for trip_cost in TripHandler.trip_costs:
            vehicle_id = trip_cost.vehicle_id
            trip_no = trip_cost.trip_no
            self.vehicle_to_trips_cost_map[vehicle_id].append(trip_cost_index)
            trip = self.trips[trip_no]
            if isinstance(trip,Trip):
                self.trip_to_vehicle_cost_map[trip_no].append(trip_cost_index)
            else:
                for sub_trip_no in trip.trips:
                    self.trip_to_vehicle_cost_map[sub_trip_no].append(trip_cost_index)
            trip_cost_index+=1

    def can_share_trips(current_time,trips,trip_nos,new_trip,current_cost,current_sequence,SHAREABLE_COST_FACTOR):
        feasible, cost, sequence = VehicleHandler.can_serve_trips(current_time,trips,new_trip,current_sequence)
        if feasible and cost <= SHAREABLE_COST_FACTOR*current_cost:
            return SharedTrip(0,trip_nos,cost,sequence)
        return None #(feasible,cost,trip_nos,current_time,trips)
    
    def process_shared_trip_result(shared_trip):
        if shared_trip != None:
            TripHandler.shared_trips_to_create.append(shared_trip)
        # else:
        #     print(shared_trip)

    def update_shared_trip_numbers(self,cardinality):
        self.selected_combinations = []
        for shared_trip in TripHandler.shared_trips_to_create:
            new_shared_trip_no = self.get_new_trip_no()
            shared_trip.number = new_shared_trip_no
            self.trips.append(shared_trip)
            self.shared_trips_map[cardinality].append(new_shared_trip_no)
            self.selected_combinations.append(shared_trip.trips)

    def generate_shared_trips(self,current_time,max_cardinality,max_num_thread,SHAREABLE_COST_FACTOR):
        cardinality = 2
        self.selected_combinations = []
        while cardinality <= max_cardinality:
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
                    pool.apply_async(TripHandler.can_share_trips,args=(current_time,trips,set(trip_nos),trip1.id,current_cost,[],SHAREABLE_COST_FACTOR,), callback=TripHandler.process_shared_trip_result)
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
                        if len(uncommon_trips) == 1:
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
                                    pool.apply_async(TripHandler.can_share_trips,args=(current_time,trips,trip_nos,trip.id,current_cost,shared_trip1.sequence,SHAREABLE_COST_FACTOR,), callback=TripHandler.process_shared_trip_result)
                pool.close()
                pool.join()
            self.update_shared_trip_numbers(cardinality)
            print("time to generate cardinal {0} trips: {1}".format(cardinality,st-time.time()))
            print("Number of cardinal {0} trips: {1}".format(cardinality,len(self.shared_trips_map[cardinality])))
            cardinality+=1
    
    def log_with_timestamp(self,timestamp,message):
        logging.info('{0}: {1}'.format(timestamp,message))

    def get_x(self,i):
        if self.x[i] > 0.9:
            return 1
        return 0
    
    def assign_trips_gurobi(self,vehicles,requests,request_bus_combinations,active_requests,penalty,current_time):
        trip_count = len(TripHandler.trip_costs)
        empty_trip_count = self.empty_trip_count
        request_count = len(requests)

        logging.debug("Started building optimization problem")
        m = gp.Model('VRP')
        var_type = GRB.BINARY
        trip_costs = np.zeros(trip_count)
        for i in range(trip_count):
            trip_costs[i] = TripHandler.trip_costs[i].cost
        x_t = m.addVars(trip_count,lb=0,ub=1,obj=trip_costs,name="t", vtype=var_type)
        x_e = m.addVars(empty_trip_count,lb=0,ub=1,name="e", vtype=var_type)
        x_r = m.addVars(request_count,lb=0,ub=1,obj=np.ones(request_count)*penalty,name="r", vtype=var_type)

        m.addConstrs((gp.quicksum(x_t[i] for i in self.vehicle_to_trips_cost_map[vehicle_id]) <= 1 for vehicle_id in list(self.vehicle_to_trips_cost_map.keys())), "veh")

        request_no = 0
        for request in requests:
            row_indices = []
            empty_indices = []
            trip_no = self.ondemand_only_trip_map[request.id]
            cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
            row_indices.extend(cost_map_indices)

            for combination_label in request_bus_combinations[request.id]:
                combination = request_bus_combinations[request.id][combination_label]
                for bus_trip in combination:
                    first_mile_trip_id = bus_trip.first_mile_trip
                    last_mile_trip_id = bus_trip.last_mile_trip
                    
                    if bus_trip.first_mile_trip_empty:
                        empty_indices.append(first_mile_trip_id)
                    else:
                        row_indices.extend(self.trip_to_vehicle_cost_map[first_mile_trip_id])

            m.addConstr(x_r[request_no]+gp.quicksum(x_e[i] for i in empty_indices)+gp.quicksum(x_t[i] for i in row_indices) == 1,"req_{0}".format(request.id))
            
            # all the previously assigned requests should be picked up
            if request.id in active_requests:
                m.addConstr(x_r[request_no] == 0,"active_req_{0}".format(request.id))
            request_no+=1

        bus_combinations_no = 0
        for request in requests:
            for combination_label in request_bus_combinations[request.id]:
                combination = request_bus_combinations[request.id][combination_label]
                for bus_trip in combination:
                    first_mile_indices = []
                    first_mile_empty_indices = []
                    last_mile_indices = []
                    last_mile_empty_indices = []
                    first_mile_trip_id = bus_trip.first_mile_trip
                    last_mile_trip_id = bus_trip.last_mile_trip
                    
                    if bus_trip.first_mile_trip_empty:
                        first_mile_empty_indices.append(first_mile_trip_id)
                    else:
                        first_mile_indices.extend(self.trip_to_vehicle_cost_map[first_mile_trip_id])

                    if bus_trip.last_mile_trip_empty:
                        last_mile_empty_indices.append(last_mile_trip_id)
                    else:
                        last_mile_indices.extend(self.trip_to_vehicle_cost_map[last_mile_trip_id])

                    m.addConstr(gp.quicksum(x_e[i] for i in first_mile_empty_indices)+gp.quicksum(x_t[i] for i in first_mile_indices) -gp.quicksum(x_e[i] for i in last_mile_empty_indices)-gp.quicksum(x_t[i] for i in last_mile_indices) == 0,"com_{0}".format(combination_label))
                    bus_combinations_no+=1
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            self.log_with_timestamp(current_time,"Total time spent on optimization: {0}".format(m.Runtime))

            self.vehicle_assignment = {}
            self.request_assignment = {}
            self.unassigned_trip_count = 0
            self.taxi_only_trip_count = 0
            self.with_one_bus_trip_count = 0
            self.with_two_bus_trip_count = 0
            self.added_distance = 0
            self.trip_sizes = []

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
                        self.request_assignment[request.id] = TaxiOnlyAssignment(vehicle_id)
                        found_assignment = True
                        self.taxi_only_trip_count+=1
                        break

                for combination_label in request_bus_combinations[request.id]:
                    if found_assignment:
                        break
                    combination = request_bus_combinations[request.id][combination_label]
                    for bus_trip in combination:
                        first_mile_trip_id = bus_trip.first_mile_trip
                        last_mile_trip_id = bus_trip.last_mile_trip
                        found_first_trip_assignment = False
                        found_last_trip_assignment = False

                        first_mile_trip_cost_index = None
                        last_mile_trip_cost_index = None
                        
                        if bus_trip.first_mile_trip_empty:
                            if x_e[first_mile_trip_id].X == 1:
                                found_first_trip_assignment = True
                        else:
                            for index in self.trip_to_vehicle_cost_map[first_mile_trip_id]:
                                if x_t[index].X == 1:
                                    found_first_trip_assignment = True
                                    first_mile_trip_cost_index = index
                                    break

                        if bus_trip.last_mile_trip_empty:
                            if x_e[last_mile_trip_id].X == 1:
                                found_last_trip_assignment = True
                        else:
                            for index in self.trip_to_vehicle_cost_map[last_mile_trip_id]:
                                if x_t[index].X == 1:
                                    found_last_trip_assignment = True
                                    last_mile_trip_cost_index = index
                                    break

                        if found_first_trip_assignment and found_last_trip_assignment:
                            self.request_assignment[request.id] = AssignmentWithBus(bus_trip)
                            if not bus_trip.first_mile_trip_empty:
                                trip_cost = TripHandler.trip_costs[first_mile_trip_cost_index]
                                vehicle_id = trip_cost.vehicle_id
                                self.request_assignment[request.id].first_mile_vehicle = vehicle_id
                            if not bus_trip.last_mile_trip_empty:
                                trip_cost = TripHandler.trip_costs[last_mile_trip_cost_index]
                                vehicle_id = trip_cost.vehicle_id
                                self.request_assignment[request.id].last_mile_vehicle = vehicle_id
                            found_assignment = True
                            if bus_trip.bus_count() == 1:
                                self.with_one_bus_trip_count += 1
                            else:
                                self.with_two_bus_trip_count += 1
                            break
                if not found_assignment:
                    self.unassigned_trip_count+=1

            logging.info('{0}: No of requests: {1}, unassigned requests: {2}, taxi only requests: {3}, requests served by busses: {4}'.format(current_time,request_count,self.unassigned_trip_count,self.taxi_only_trip_count,self.with_one_bus_trip_count+self.with_two_bus_trip_count))

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

            self.rebalancing_assignment = {}
            if m.Status == GRB.OPTIMAL:
                for i in range(number_of_vehicles):
                    for j in range(number_of_requests):
                        if y_vr[i,j].X == 1:
                            vehicle_id = empty_vehicles[i]
                            origin = unassigned_requests[j].origin
                            self.rebalancing_assignment[vehicle_id] = origin
                            break

    def assign_trips(self,vehicles,requests,request_bus_combinations,penalty,current_time):
        trip_count = len(TripHandler.trip_costs)
        empty_trip_count = self.empty_trip_count
        request_count = len(requests)
        vehicle_count = len(vehicles)
        numvar = trip_count+empty_trip_count+request_count
        numcon = self.bus_combinations+vehicle_count+request_count
        x = np.zeros(numvar)


        logging.debug("Started building optimization problem")

        with mosek.Env() as env:
            with env.Task(0, 1) as task:
                task.appendvars(numvar)

                for j in range(trip_count):
                    task.putcj(j, TripHandler.trip_costs[j].cost)
                    task.putvarbound(j, mosek.boundkey.ra, 0, 1)

                for j in range(trip_count,trip_count+empty_trip_count):
                    task.putcj(j, 0)
                    task.putvarbound(j, mosek.boundkey.ra, 0, 1)

                for j in range(trip_count+empty_trip_count,numvar):
                    task.putcj(j, penalty)
                    task.putvarbound(j, mosek.boundkey.ra, 0, 1)

                task.appendcons(numcon)
    
                vehicle_no = 0
                for vehicle_id in self.vehicle_to_trips_cost_map:
                    task.putconbound(vehicle_no, mosek.boundkey.up, 0, 1)
                    trips = self.vehicle_to_trips_cost_map[vehicle_id]
                    task.putarow(vehicle_no,trips,[1]*len(trips))
                    vehicle_no+=1

                request_no = 0
                for request in requests:
                    task.putconbound(vehicle_count+request_no, mosek.boundkey.fx, 1, 1)
                    row_indices = [trip_count+empty_trip_count+request_no]
                    row_values = [1]
                    trip_no = self.ondemand_only_trip_map[request.id]
                    cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
                    row_indices.extend(cost_map_indices)
                    row_values.extend([1]*len(cost_map_indices))

                    for combination_label in request_bus_combinations[request.id]:
                        combination = request_bus_combinations[request.id][combination_label]
                        for bus_trip in combination:
                            first_mile_trip_id = bus_trip.first_mile_trip
                            last_mile_trip_id = bus_trip.last_mile_trip
                            
                            if bus_trip.first_mile_trip_empty:
                                row_indices.append(first_mile_trip_id+trip_count)
                                row_values.append(1)
                            else:
                                row_indices.extend(self.trip_to_vehicle_cost_map[first_mile_trip_id])
                                row_values.extend([1]*len(self.trip_to_vehicle_cost_map[first_mile_trip_id]))

                    task.putarow(request_no+vehicle_count,row_indices,row_values)
                    request_no+=1

                bus_combinations_no = 0
                for request in requests:
                    for combination_label in request_bus_combinations[request.id]:
                        combination = request_bus_combinations[request.id][combination_label]
                        for bus_trip in combination:
                            task.putconbound(vehicle_count+request_count+bus_combinations_no, mosek.boundkey.fx, 0, 0)
                            row_indices = []
                            row_values = []
                            first_mile_trip_id = bus_trip.first_mile_trip
                            last_mile_trip_id = bus_trip.last_mile_trip
                            
                            if bus_trip.first_mile_trip_empty:
                                row_indices.append(first_mile_trip_id+trip_count)
                                row_values.append(1)
                            else:
                                row_indices.extend(self.trip_to_vehicle_cost_map[first_mile_trip_id])
                                row_values.extend([1]*len(self.trip_to_vehicle_cost_map[first_mile_trip_id]))

                            if bus_trip.last_mile_trip_empty:
                                row_indices.append(last_mile_trip_id+trip_count)
                                row_values.append(-1)
                            else:
                                row_indices.extend(self.trip_to_vehicle_cost_map[last_mile_trip_id])
                                row_values.extend([-1]*len(self.trip_to_vehicle_cost_map[last_mile_trip_id]))

                            task.putarow(vehicle_count+request_count+bus_combinations_no,row_indices,row_values)
                            bus_combinations_no+=1

                task.putobjsense(mosek.objsense.minimize)
                task.putvartypelist(np.arange(numvar),
                            [mosek.variabletype.type_int]*numvar)
                task.putdouparam(mosek.dparam.mio_max_time, self.ipm_solver_timeout)

                logging.debug("Started optimization")

                task.optimize()

                logging.debug("Finished optimization")
                task.getxx(mosek.soltype.itg, x)

                task.writedata("data.opf")
                
                prosta = task.getprosta(mosek.soltype.itg)
                solsta = task.getsolsta(mosek.soltype.itg)
                message = None
                if solsta in [mosek.solsta.integer_optimal]:
                    message = "Optimal solution"
                elif solsta == mosek.solsta.prim_feas:
                    message = "Feasible solution"
                elif mosek.solsta.unknown:
                    if prosta == mosek.prosta.prim_infeas_or_unbounded:
                        message = "Problem status Infeasible or unbounded."
                    elif prosta == mosek.prosta.prim_infeas:
                        message = "Problem status Infeasible."
                    elif prosta == mosek.prosta.unkown:
                        message = "Problem status unkown."
                    else:
                        message = "Other problem status."
                else:
                    message = "Other solution status"
                time_spent = task.getdouinf(mosek.dinfitem.optimizer_time)
                self.log_with_timestamp(current_time,"{0}, Total time spent on optimization: {1}".format(message,time_spent))

        self.x = x
        self.vehicle_assignment = {}
        self.request_assignment = {}
        self.unassigned_trip_count = 0
        self.taxi_only_trip_count = 0
        self.with_one_bus_trip_count = 0
        self.with_two_bus_trip_count = 0
        self.added_distance = 0
        self.trip_sizes = []

        for vehicle_id in self.vehicle_to_trips_cost_map:
            for i in self.vehicle_to_trips_cost_map[vehicle_id]:
                if self.get_x(i) == 1:
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
                if self.get_x(index) == 1:
                    trip_cost = TripHandler.trip_costs[index]
                    vehicle_id = trip_cost.vehicle_id
                    self.request_assignment[request.id] = TaxiOnlyAssignment(vehicle_id)
                    found_assignment = True
                    self.taxi_only_trip_count+=1
                    break

            for combination_label in request_bus_combinations[request.id]:
                if found_assignment:
                    break
                combination = request_bus_combinations[request.id][combination_label]
                for bus_trip in combination:
                    first_mile_trip_id = bus_trip.first_mile_trip
                    last_mile_trip_id = bus_trip.last_mile_trip
                    found_first_trip_assignment = False
                    found_last_trip_assignment = False

                    first_mile_trip_cost_index = None
                    last_mile_trip_cost_index = None
                    
                    if bus_trip.first_mile_trip_empty:
                        if self.get_x(first_mile_trip_id+trip_count) == 1:
                            found_first_trip_assignment = True
                    else:
                        for index in self.trip_to_vehicle_cost_map[first_mile_trip_id]:
                            if self.get_x(index) == 1:
                                found_first_trip_assignment = True
                                first_mile_trip_cost_index = index
                                break

                    if bus_trip.last_mile_trip_empty:
                        if self.get_x(last_mile_trip_id+trip_count) == 1:
                            found_last_trip_assignment = True
                    else:
                        for index in self.trip_to_vehicle_cost_map[last_mile_trip_id]:
                            if self.get_x(index) == 1:
                                found_last_trip_assignment = True
                                last_mile_trip_cost_index = index
                                break

                    if found_first_trip_assignment and found_last_trip_assignment:
                        self.request_assignment[request.id] = AssignmentWithBus(bus_trip)
                        if not bus_trip.first_mile_trip_empty:
                            trip_cost = TripHandler.trip_costs[first_mile_trip_cost_index]
                            vehicle_id = trip_cost.vehicle_id
                            self.request_assignment[request.id].first_mile_vehicle = vehicle_id
                        if not bus_trip.last_mile_trip_empty:
                            trip_cost = TripHandler.trip_costs[last_mile_trip_cost_index]
                            vehicle_id = trip_cost.vehicle_id
                            self.request_assignment[request.id].last_mile_vehicle = vehicle_id
                        found_assignment = True
                        if bus_trip.bus_count() == 1:
                            self.with_one_bus_trip_count += 1
                        else:
                            self.with_two_bus_trip_count += 1
                        break
            if not found_assignment:
                self.unassigned_trip_count+=1

        # Logging
        for request in requests:
            trip_no = self.ondemand_only_trip_map[request.id]
            cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
            no_bus_trips = 0
            first_mile_trips = []
            last_mile_trips = []
            for combination_label in request_bus_combinations[request.id]:
                combination = request_bus_combinations[request.id][combination_label]
                no_bus_trips += len(combination)
                for bus_trip in combination:
                    first_mile_trip_id = bus_trip.first_mile_trip
                    last_mile_trip_id = bus_trip.last_mile_trip

                    first_mile_trip_cost_index = None
                    last_mile_trip_cost_index = None
                    
                    if bus_trip.first_mile_trip_empty:
                        first_mile_trips.append(-1)
                    else:
                        first_mile_trips.append(len(self.trip_to_vehicle_cost_map[first_mile_trip_id]))

                    if bus_trip.last_mile_trip_empty:
                        last_mile_trips.append(-1)
                    else:
                        last_mile_trips.append(len(self.trip_to_vehicle_cost_map[last_mile_trip_id]))

            last_mile_trips_copy = [str(i) for i in last_mile_trips]
            first_mile_trips_copy = [str(i) for i in first_mile_trips]
            logging.info('Requests ID: {0}, direct vehicles: {1}, no of bus trips: {2}, first mile vehicles: {3}, last mile vehicles: {4}'.format(request.id,len(cost_map_indices),no_bus_trips,",".join(first_mile_trips_copy),",".join(last_mile_trips_copy)))
        logging.info('{0}: No of requests: {1}, unassigned requests: {2}, taxi only requests: {3}, requests served by busses: {4}'.format(current_time,request_count,self.unassigned_trip_count,self.taxi_only_trip_count,self.with_one_bus_trip_count+self.with_two_bus_trip_count))
