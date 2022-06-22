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

class TripHandler:
    def __init__(self,current_time,vehicles,requests,request_bus_combinations,distance_cutoff,ipm_solver_timeout,penalty,MAX_CARDINALITY,MAX_THREAD_CNT,SHAREABLE_COST_FACTOR):
        self.trips = []
        self.shared_trips_map = {}
        self.empty_trip_count = 0
        self.vehicle_only_trip_map = {}
        self.ipm_solver_timeout = ipm_solver_timeout
        self.walk_distance_cutoff = distance_cutoff
        self.vehicle_to_trips_cost_map = {}
        self.trip_to_vehicle_cost_map = {}
        self.SHAREABLE_COST_FACTOR = SHAREABLE_COST_FACTOR
        self.generate_vehicle_only_trips(requests,current_time)
        self.generate_trips_with_bus(requests,request_bus_combinations)
        # self.generate_shared_trips(current_time,MAX_CARDINALITY)
        self.generate_trip_costs(vehicles,current_time,MAX_THREAD_CNT)
        self.assign_trips(vehicles,requests,request_bus_combinations,penalty,current_time)

    def get_new_trip_no(self):
        return len(self.trips)

    def get_trip_cost(self,origin,destination):
        return NetworkHandler.travel_distance(origin,destination)
    
    def generate_vehicle_only_trips(self,requests,current_time):
        for request in requests:
            origin = request.origin
            destination = request.destination
            trip = self.create_trip(request,origin,destination,current_time, request.arrival_time,allow_walk=False)
            self.trips.append(trip)
            self.vehicle_only_trip_map[request.id] = trip.number

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

    def create_trip(self,request,origin,destination,pick_up_time,arrival_time,bus_combination=None,first_last_mile_type=None,allow_walk=True):
        if allow_walk and self.can_walk(origin,destination):
            return None
        trip_no = self.get_new_trip_no()
        cost = self.get_trip_cost(origin,destination)
        return Trip(request.id,trip_no,pick_up_time, arrival_time, origin, destination,cost,bus_combination=bus_combination,first_last_mile_type=first_last_mile_type)
    
    def get_first_mile_trip(self,request,bustrip):
        origin = request.origin
        destination = bustrip.pick_up_stop
        return self.create_trip(request,origin,destination,request.pick_up_time, bustrip.leaving_time,bus_combination=bustrip.id,first_last_mile_type=0)

    def get_last_mile_trip(self,request,bustrip):
        destination = request.destination
        origin = bustrip.destination_stop
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

    def can_share_trips(self,current_time,trip_nos,current_cost):
        trips = {}
        for trip_no in trip_nos:
            trip = self.trips[trip_no]
            trips[trip.id] = trip
        feasible, cost = VehicleHandler.can_serve_trips(current_time,trips)
        if feasible and cost <= self.SHAREABLE_COST_FACTOR*current_cost:
            return feasible, cost
        return False, cost

    def do_trips_conflict(self,trip_nos):
        for trip_no1 in trip_nos:
            trip1 = self.trips[trip_no1]
            for trip_no2 in trip_nos:
                if trip_no1 != trip_no2:
                    trip2 = self.trips[trip_no2]
                    if trip1.request_id == trip2.request_id:
                        if trip1.bus_combination != trip2.bus_combination or trip1.first_last_mile_type == trip2.first_last_mile_type:
                            return True
        return False

    def generate_shared_trips(self,current_time,max_cardinality):
        cardinality = 2
        while cardinality <= max_cardinality:
            self.shared_trips_map[cardinality] = []
            if cardinality == 2:
                no_of_trips = len(self.trips)
                for trip_nos in itertools.combinations(list(range(no_of_trips)),cardinality):
                    trip1 = self.trips[trip_nos[0]]
                    trip2 = self.trips[trip_nos[1]]
                    if trip1.number != trip2.number and (not self.do_trips_conflict(trip_nos)):
                        current_cost = trip1.cost+trip2.cost
                        shareable, cost = self.can_share_trips(current_time,trip_nos,current_cost)
                        if shareable:
                            new_shared_trip_no = self.get_new_trip_no()
                            self.trips.append(SharedTrip(new_shared_trip_no,trip_nos,cost))
                            self.shared_trips_map[cardinality].append(new_shared_trip_no)
            else:
                tried_combinations = []
                for shared_trip_no in self.shared_trips_map[cardinality-1]:
                    shared_trip = self.trips[shared_trip_no]
                    for trip in self.trips:
                        if trip.number not in shared_trip.trips:
                            current_cost = trip.cost+shared_trip.cost
                            trips = shared_trip.trips.copy().append(trip.number)
                            if not self.do_trips_conflict(trips):
                                combination_already_tested = False
                                for tried_combination in tried_combinations:
                                    if tried_combination == set(trips):
                                        combination_already_tested = True
                                        break
                                if not combination_already_tested:
                                    tried_combinations.append(set(trips))
                                    sub_combination_found = False
                                    for combination in itertools.combinations(trips,cardinality-1):
                                        sub_combination_found = True
                                        for sub_shared_trip_no in self.shared_trips_map[cardinality-1]:
                                            sub_shared_trip = self.trips[sub_shared_trip_no]
                                            if set(sub_shared_trip.trips) == set(combination):
                                                sub_combination_found = True
                                        if not sub_combination_found:
                                            break
                                    shareabile, cost= self.can_share_trips(current_time,trips,current_cost)
                                    if sub_combination_found and shareabile:
                                        new_shared_trip_no = self.get_new_trip_no()
                                        self.trips.append(SharedTrip(new_shared_trip_no,trips,cost))
                                        self.shared_trips_map[cardinality].append(new_shared_trip_no)
            cardinality+=1
    
    def log_with_timestamp(self,timestamp,message):
        logging.info('{0}: {1}'.format(timestamp,message))

    def get_x(self,i):
        if self.x[i] > 0.9:
            return 1
        return 0

    def assign_trips(self,vehicles,requests,request_bus_combinations,penalty,current_time):
        trip_count = len(TripHandler.trip_costs)
        empty_trip_count = self.empty_trip_count
        request_count = len(requests)
        vehicle_count = len(vehicles)
        numvar = trip_count+empty_trip_count+request_count
        numcon = self.bus_combinations+vehicle_count+request_count
        x = np.zeros(numvar)

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
                    trip_no = self.vehicle_only_trip_map[request.id]
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

                task.optimize()
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
            trip_no = self.vehicle_only_trip_map[request.id]
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

        logging.info('{0}: No of requests: {1}, unassigned requests: {2}, taxi only requests: {3}, requests served by busses: {4}'.format(current_time,request_count,self.unassigned_trip_count,self.taxi_only_trip_count,self.with_one_bus_trip_count+self.with_two_bus_trip_count))
