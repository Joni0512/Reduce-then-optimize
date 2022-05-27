from structure.trip import Trip
from structure.shared_trip import SharedTrip
from structure.assignment import AssignmentWithBus
from structure.assignment import TaxiOnlyAssignment
import numpy as np
import mosek
import logging
import itertools

class TripHandler:
    def __init__(self,current_time,network_handler,vehicle_handler,requests,request_bus_combinations,speed,distance_cutoff,ipm_solver_timeout,penalty,MAX_CARDINALITY,SHAREABLE_COST_FACTOR):
        self.trips = []
        self.shared_trips_map = {}
        self.empty_trips = []
        self.vehicle_only_trip_map = {}
        self.ipm_solver_timeout = ipm_solver_timeout
        self.walk_distance_cutoff = distance_cutoff/speed
        self.vehicle_trips_cost = []
        self.vehicle_to_trips_cost_map = {}
        self.reverse_vehicle_to_trips_cost_map = []
        self.reverse_trip_to_trips_cost_map = []
        self.trip_to_vehicle_cost_map = {}
        self.SHAREABLE_COST_FACTOR = SHAREABLE_COST_FACTOR
        self.generate_vehicle_only_trips(requests,network_handler,vehicle_handler)
        self.generate_trips_with_bus(network_handler,vehicle_handler,requests,request_bus_combinations)
        self.generate_shared_trips(network_handler,vehicle_handler,current_time,MAX_CARDINALITY)
        self.generate_trip_costs(network_handler,vehicle_handler,current_time)
        self.assign_trips(vehicle_handler,requests,request_bus_combinations,penalty,current_time)

    def get_new_trip_no(self):
        return len(self.trips)

    def get_trip_cost(self,network_handler,vehicle_handler,origin,destination):
        return vehicle_handler.get_cost_of_travel(network_handler.travel_time(origin,destination))
    
    def generate_vehicle_only_trips(self,requests,network_handler,vehicle_handler):
        for request in requests:
            trip_no = self.get_new_trip_no()
            self.vehicle_only_trip_map[request.id] = trip_no
            cost = self.get_trip_cost(network_handler,vehicle_handler,request.origin, request.destination)
            self.trips.append(Trip(request.id,request.id,trip_no,request.pick_up_time, request.arrival_time, request.origin, request.destination, cost))

    def generate_trips_with_bus(self,network_handler,vehicle_handler,requests,request_bus_combinations):
        self.bus_combinations = 0
        for request in requests:
            if request.id in request_bus_combinations:
                for bus_combination in request_bus_combinations[request.id]:
                    combination = request_bus_combinations[request.id][bus_combination]
                    for bus_trip in combination:
                        self.bus_combinations+=1
                        first_mile_trip = self.get_first_mile_trip(network_handler,vehicle_handler,request,bus_trip)
                        if first_mile_trip == None:
                            bus_trip.first_mile_trip = len(self.empty_trips)
                            self.empty_trips.append(first_mile_trip)
                            bus_trip.first_mile_trip_empty = True
                        else:
                            bus_trip.first_mile_trip = first_mile_trip.number
                            self.trips.append(first_mile_trip)

                        last_mile_trip = self.get_last_mile_trip(network_handler,vehicle_handler,request,bus_trip)
                        if last_mile_trip == None:
                            bus_trip.last_mile_trip = len(self.empty_trips)
                            self.empty_trips.append(last_mile_trip)
                            bus_trip.last_mile_trip_empty = True
                        else:
                            bus_trip.last_mile_trip = last_mile_trip.number
                            self.trips.append(last_mile_trip)
    
    def get_first_mile_trip(self,network_handler,vehicle_handler,request,bustrip):
        origin = request.origin
        destination = bustrip.pick_up_stop
        trip_no = self.get_new_trip_no()
        cost = self.get_trip_cost(network_handler,vehicle_handler,origin,destination)
        if self.can_walk(network_handler,origin,destination):
            return None
        return Trip("{0}:{1}:F".format(request.id,bustrip.id),request.id,trip_no,request.pick_up_time, bustrip.leaving_time, origin, destination,cost)

    def get_last_mile_trip(self,network_handler,vehicle_handler,request,bustrip):
        destination = request.destination
        origin = bustrip.destination_stop
        trip_no = self.get_new_trip_no()
        cost = self.get_trip_cost(network_handler,vehicle_handler,origin,destination)
        if self.can_walk(network_handler,origin,destination):
            return None
        return Trip("{0}:{1}:L".format(request.id,bustrip.id),request.id,trip_no,bustrip.arrival_time, request.arrival_time, origin, destination, cost)
    
    def can_walk(self,network_handler,origin,destination):
        distance = network_handler.travel_time(origin,destination)
        return distance <= self.walk_distance_cutoff

    def generate_trip_costs(self,network_handler,vehicle_handler,current_time):
        for vehicle_id in vehicle_handler.vehicles:
            vehicle = vehicle_handler.vehicles[vehicle_id]
            self.vehicle_to_trips_cost_map[vehicle_id] = []
            for trip in self.trips:
                if isinstance(trip,Trip):
                    trip_no = trip.number
                    if trip_no not in self.trip_to_vehicle_cost_map:
                        self.trip_to_vehicle_cost_map[trip_no] = []
                    added_cost, feasibility = vehicle_handler.add_new_trips(network_handler,current_time, vehicle, [trip], add=False)
                    if feasibility:
                        self.vehicle_to_trips_cost_map[vehicle_id].append(len(self.vehicle_trips_cost))
                        self.trip_to_vehicle_cost_map[trip_no].append(len(self.vehicle_trips_cost))
                        self.vehicle_trips_cost.append(added_cost)
                        self.reverse_vehicle_to_trips_cost_map.append(vehicle_id)
                        self.reverse_trip_to_trips_cost_map.append(trip_no)
                else:
                    shared_trip = trip
                    trips = []
                    for sub_trip_no in shared_trip.trips:
                        trips.append(self.trips[sub_trip_no])
                    added_cost, feasibility = vehicle_handler.add_new_trips(network_handler,current_time, vehicle, trips, add=False)
                    if feasibility:
                        self.vehicle_to_trips_cost_map[vehicle_id].append(len(self.vehicle_trips_cost))
                        for sub_trip_no in shared_trip.trips:
                            self.trip_to_vehicle_cost_map[sub_trip_no].append(len(self.vehicle_trips_cost))
                        self.vehicle_trips_cost.append(added_cost)
                        self.reverse_vehicle_to_trips_cost_map.append(vehicle_id)
                        self.reverse_trip_to_trips_cost_map.append(shared_trip.number)


    def can_share_trips(self,network_handler,vehicle_handler,current_time,trip_nos,current_cost):
        trips = {}
        for trip_no in trip_nos:
            trip = self.trips[trip_no]
            trips[trip.id] = trip
        feasible, cost = vehicle_handler.can_serve_trips(network_handler,current_time,trips)
        if feasible and cost <= self.SHAREABLE_COST_FACTOR*current_cost:
            return feasible, cost
        return False, cost

    def generate_shared_trips(self,network_handler,vehicle_handler,current_time,max_cardinality):
        cardinality = 2
        while cardinality <= max_cardinality:
            self.shared_trips_map[cardinality] = []
            if cardinality == 2:
                for trip_nos in itertools.combinations(list(range(len(self.trips))),cardinality):
                    trip1 = self.trips[trip_nos[0]]
                    trip2 = self.trips[trip_nos[1]]
                    if trip1.number != trip2.number and trip1.request_id != trip2.request_id:
                        current_cost = trip1.cost+trip2.cost
                        shareable, cost = self.can_share_trips(network_handler,vehicle_handler,current_time,trip_nos,current_cost)
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
                                shareabile, cost= self.can_share_trips(network_handler,vehicle_handler,current_time,trips,current_cost)
                                if sub_combination_found and shareabile:
                                    new_shared_trip_no = self.get_new_trip_no()
                                    self.trips.append(SharedTrip(new_shared_trip_no,trips,cost))
                                    self.shared_trips_map[cardinality].append(new_shared_trip_no)
            cardinality+=1
    
    def log_with_timestamp(self,timestamp,message):
        logging.info('{0}: {1}'.format(timestamp,message))

    def assign_trips(self,vehicle_handler,requests,request_bus_combinations,penalty,current_time):
        trip_count = len(self.vehicle_trips_cost)
        empty_trip_count = len(self.empty_trips)
        request_count = len(requests)
        c = np.array(self.vehicle_trips_cost)
        vehicle_count = len(vehicle_handler.vehicles)
        numvar = trip_count+empty_trip_count+request_count
        numcon = self.bus_combinations+vehicle_count+request_count
        x = np.zeros(numvar)

        with mosek.Env() as env:
            with env.Task(0, 1) as task:
                task.appendvars(numvar)

                for j in range(trip_count):
                    task.putcj(j, c[j])
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
        self.with_bus_trip_count = 0

        for vehicle_id in self.vehicle_to_trips_cost_map:
            for i in self.vehicle_to_trips_cost_map[vehicle_id]:
                if x[i] == 1:
                    trip_no = self.reverse_trip_to_trips_cost_map[i]
                    trip = self.trips[trip_no]
                    trips = []
                    if isinstance(trip,Trip):
                        trips.append(trip)
                    else:
                        for sub_trip_no in trip.trips:
                            trips.append(self.trips[sub_trip_no])
                    self.vehicle_assignment[vehicle_id] = trips

        for request in requests:
            found_assignment = False
            trip_no = self.vehicle_only_trip_map[request.id]
            cost_map_indices = self.trip_to_vehicle_cost_map[trip_no]
            for index in cost_map_indices:
                if x[index] == 1:
                    vehicle_id = self.reverse_vehicle_to_trips_cost_map[index]
                    # self.vehicle_assignment[vehicle_id] = self.trips[trip_no]
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
                        if x[first_mile_trip_id+trip_count] == 1:
                            found_first_trip_assignment = True
                    else:
                        for index in self.trip_to_vehicle_cost_map[first_mile_trip_id]:
                            if x[index] == 1:
                                found_first_trip_assignment = True
                                first_mile_trip_cost_index = index
                                break

                    if bus_trip.last_mile_trip_empty:
                        if x[last_mile_trip_id+trip_count] == 1:
                            found_last_trip_assignment = True
                    else:
                        for index in self.trip_to_vehicle_cost_map[last_mile_trip_id]:
                            if x[index] == 1:
                                found_last_trip_assignment = True
                                last_mile_trip_cost_index = index
                                break

                    if found_first_trip_assignment and found_last_trip_assignment:
                        self.request_assignment[request.id] = AssignmentWithBus(bus_trip)
                        if not bus_trip.first_mile_trip_empty:
                            vehicle_id = self.reverse_vehicle_to_trips_cost_map[first_mile_trip_cost_index]
                            # self.vehicle_assignment[vehicle_id] = self.trips[first_mile_trip_id]
                            self.request_assignment[request.id].first_mile_vehicle = vehicle_id
                        if not bus_trip.last_mile_trip_empty:
                            vehicle_id = self.reverse_vehicle_to_trips_cost_map[last_mile_trip_cost_index]
                            # self.vehicle_assignment[vehicle_id] = self.trips[last_mile_trip_id]
                            self.request_assignment[request.id].last_mile_vehicle = vehicle_id
                        found_assignment = True
                        self.with_bus_trip_count+=1
                        break
            if not found_assignment:
                self.unassigned_trip_count+=1

        logging.info('{0}: No of requests: {1}, unassigned requests: {2}, taxi only requests: {3}, requests served by busses: {4}'.format(current_time,request_count,self.unassigned_trip_count,self.taxi_only_trip_count,self.with_bus_trip_count))
