import pandas as pd
import pickle

from datetime import datetime, timedelta

from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.trip_insertion_plan import TripInsertionPlan
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.node import Node
from rtv_solver.structure.sequence import StopSequence
from rtv_solver.structure.config import Config
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.request import Request

from rtv_solver.structure.driver_run import ManifestEntry

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class VehicleHandler:
    MAX_AM_CAPACITY = 0
    MAX_WC_CAPACITY = 0
    LARGEST_TSP = 0
    
    def __init__(self, depot, driver_runs, config: Config):
        self.vehicles: dict[int, Vehicle] = {}
        self.count = 0
        self.earliest_start_time = None
        self._load_vehicles(depot, driver_runs)
        self.config = config
        VehicleHandler.LARGEST_TSP = config.LARGEST_TSP

        console_logger.info(f'{self.count} vehicle(s) in operations')

    def _load_vehicles(self, depot, driver_runs):
        """
        Load all vehicles and initialize their location at the depot

        # TODO remove vehicles that are not active anymore based on the current_time of the iteration in relation to their end_time; how should the behaviour look like
        """
        start_location = depot.copy() # requires copy, otherwise they all point to the same dictionary of the depot-dict

        for driver_run in driver_runs:
            vehicle_data = self._extract_vehicle_state(driver_run)

            vehicle_id = int(vehicle_data[PayloadParser.DRIVER_STATE_RUN_ID])
            am_capacity = int(vehicle_data[PayloadParser.DRIVER_STATE_AM_CAP])
            wc_capacity = int(vehicle_data[PayloadParser.DRIVER_STATE_WC_CAP])
            start_time = vehicle_data[PayloadParser.DRIVER_STATE_START_TIME]
            end_time = vehicle_data[PayloadParser.DRIVER_STATE_END_TIME]

            manifest = self._extract_vehicle_manifest(driver_run)

            vehicle = Vehicle(
                vehicle_id,
                start_location, # FIXME? this resets the location of the vehicles for each iteration; this does not seem to be intended behaviour or we must update it in a next step
                am_capacity,
                wc_capacity,
                start_time,
                end_time,
                start_location,
                manifest
            )

            self.vehicles[vehicle_id] = vehicle
            self.count += 1
            
            self._update_max_capacities(am_capacity, wc_capacity)
            self._update_earliest_start_time(start_time)

    @staticmethod
    def _extract_vehicle_state(driver_run):
        # depending on the definition of the payload (with or without 'state' key), this handles both cases
        if PayloadParser.DRIVER_STATE in driver_run:
            return driver_run[PayloadParser.DRIVER_STATE]
        return driver_run
    
    @staticmethod
    def _extract_vehicle_manifest(driver_run):
        if PayloadParser.DRIVER_MANIFEST in driver_run:
            manifest = [ManifestEntry.from_dict(entry) for entry in driver_run[PayloadParser.DRIVER_MANIFEST]]
            return manifest
        return []

    @staticmethod
    def _update_max_capacities(am_capacity, wc_capacity):
        VehicleHandler.MAX_AM_CAPACITY = max(VehicleHandler.MAX_AM_CAPACITY, am_capacity)
        VehicleHandler.MAX_WC_CAPACITY = max(VehicleHandler.MAX_WC_CAPACITY, wc_capacity)

    def _update_earliest_start_time(self, start_time):
        if (self.earliest_start_time is None
            or start_time < self.earliest_start_time):
            self.earliest_start_time = start_time

    @staticmethod
    def get_current_location_time(vehicle):
        next_immediate_node = vehicle.last_node 
        time_at_next_immediate_node = vehicle.time_at_last
        if len(vehicle.stop_sequence) > 0:
            time_at_next_immediate_node = vehicle.time_at_next_immediate_node
            next_immediate_node = vehicle.next_immediate_node
        return next_immediate_node, time_at_next_immediate_node

    def update_run(self, driver_run: dict) -> dict:
        """
        Update manifest of a driver_run by keeping all already-served stops and regenerating the remaining stops from the vehicle's stop_sequence.
        Returns a new driver_run dict (state + manifest).
        """
        # retrieve old information
        state = driver_run[PayloadParser.DRIVER_STATE]
        old_manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
        current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        
        vehicle_id = state[PayloadParser.DRIVER_STATE_RUN_ID]
        vehicle = self.vehicles[vehicle_id]

        # TODO what is the difference on vehicle.time_at_next and vehicle.time_at_next_immediate_node???    
        new_manifest = old_manifest[:current_order]                  
        added_manifest = self.get_manifest(vehicle, current_order)
        new_manifest.extend(added_manifest)
        # Update state meta info
        new_state = state.copy()
        new_state[PayloadParser.DRIVER_STATE_T_LOCS] = len(new_manifest)
        # Build new driver run from both parts
        new_driver_run = {PayloadParser.DRIVER_STATE: new_state, 
                          PayloadParser.DRIVER_MANIFEST: new_manifest}
        return new_driver_run

    @staticmethod
    def get_manifest(vehicle: Vehicle, current_order: int):
        """
        from current_order, build new manifest
        """
        manifest = []
        last_node, time_at_last_node = VehicleHandler.get_current_location_time(vehicle)
        for vehicle_stop in vehicle.stop_sequence:
            # default values
            trip = vehicle.trips[vehicle_stop.trip_id] # TODO breaks depot-return as that is not its own trip
            node = vehicle_stop.node
            action = VehicleStop.ACT_DROPOFF # default
            time_window_start = trip.earliest_arrival_time
            time_window_end = trip.latest_arrival_time
            dwell = trip.dwell_alight
            # update defaults if it is a PICKUP-stop
            if vehicle_stop.type == VehicleStop.ACT_PICKUP:
                action = VehicleStop.ACT_PICKUP
                time_window_start = trip.pick_up_time
                time_window_end = trip.latest_pick_up_time
                dwell = trip.dwell_pickup
            elif vehicle_stop.type == VehicleStop.ACT_DEPOT:
                action = VehicleStop.ACT_DEPOT
                time_window_start = trip.earliest_arrival_time
                time_window_end = trip.latest_arrival_time
                dwell = trip.dwell_pickup
            current_order += 1 # increment order in manifest
            stop_time = time_at_last_node + NetworkHandler.travel_time(last_node, node)
            if stop_time <= time_window_start:
                stop_time = time_window_start
            stop = {
                PayloadParser.MANIFEST_RUN_ID: vehicle.id, 
                PayloadParser.MANIFEST_BOOKING_ID: trip.request_id, 
                PayloadParser.MANIFEST_ORDER: current_order, 
                PayloadParser.MANIFEST_ACTION: action, 
                PayloadParser.MANIFEST_LOC: {
                    'lat': node.lat, 
                    'lon': node.lon, 
                    'node_id': node.id},
                PayloadParser.MANIFEST_SCHED_TIME: stop_time, # arrival time at stop
                PayloadParser.MANIFEST_AMBULATORY: trip.am_capacity, 
                PayloadParser.MANIFEST_WHEELCHAIR: trip.wc_capacity, 
                PayloadParser.MANIFEST_TIME_WINDOW_START: time_window_start, 
                PayloadParser.MANIFEST_TIME_WINDOW_END: time_window_end}
            manifest.append(stop)
            # local update for vehicle state to create complete manifest over next iteration
            last_node, time_at_last_node = node, stop_time + dwell 

        return manifest
  
    def add_manifest_to_vehicles(self, driver_runs, boarded_requests, boarded_trips, dwell_alight, dwell_pickup):
        """
        iterate over all manifests to update each vehicle based on its manifest and previously boarded requests
        """
        vehicle_ids = list(self.vehicles.keys())
        for vehicle_id in vehicle_ids:
            vehicle = self.vehicles[vehicle_id]
            driver_run = None
            for run in driver_runs:
                if int(run[PayloadParser.DRIVER_STATE][PayloadParser.DRIVER_STATE_RUN_ID]) == vehicle_id:
                    driver_run = run 
                    break # select the right driver_run
            self.add_manifest_to_vehicle(vehicle, driver_run, boarded_requests, boarded_trips, dwell_alight, dwell_pickup)
    
    @staticmethod
    def add_manifest_to_vehicle(vehicle: Vehicle, driver_run: dict, boarded_requests: dict[int, Request], boarded_trips: list[Trip], dwell_alight, dwell_pickup):
        """
        translate the manifest of a vehicle into its current position and update the vehicle accordingly
        
        TODO move this to the vehicle object in order to collect everything there
        """
        # retrieve information from dictionary
        state = driver_run[PayloadParser.DRIVER_STATE]
        state_loc = state[PayloadParser.DRIVER_STATE_LOC]
        current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        vehicle.started = True
        # retrieves next position and time there # TODO does this work as we always initialize the vehicleLocation as depot
        time_at_next_immediate_node = state[PayloadParser.DRIVER_STATE_DT_SEC]
        next_immediate_node = NetworkHandler.get_node_from_manifest_location(
            state_loc, 
            node_id = NetworkHandler.get_next_node_id(
                state_loc['lat'], 
                state_loc['lon']))

        # iterate over manifest to update all variables
        manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
        if len(manifest) > 0:
            for stop in manifest:
                if stop[PayloadParser.MANIFEST_ORDER] > current_order:
                    # future stops are not handled in this method
                    break
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    vehicle.am_capacity -= stop[PayloadParser.MANIFEST_AMBULATORY]
                    vehicle.wc_capacity -= stop[PayloadParser.MANIFEST_WHEELCHAIR]
                else: # DROPOFF
                    vehicle.am_capacity += stop[PayloadParser.MANIFEST_AMBULATORY]
                    vehicle.wc_capacity += stop[PayloadParser.MANIFEST_WHEELCHAIR]
            
            # filters remaining DROPOFFs for boarded requests (only dropoffs need to be considered as stops, PICKUPs is not required because then it is not boarded)
            filtered_manifest = []
            for stop in manifest:
                booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                if booking_id in boarded_requests and stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_DROPOFF:
                    filtered_manifest.append(stop)

            for stop in filtered_manifest:
                # find boarded 
                trip_of_stop = None
                for trip in boarded_trips:
                    if trip.request_id == stop[PayloadParser.MANIFEST_BOOKING_ID]:
                        trip_of_stop = trip
                        break
                # add trip to vehicle
                vehicle.trips[trip_of_stop.id] = trip_of_stop
                vehicle.picked.append(trip_of_stop.id)
                vehicle_stop = VehicleStop(
                    trip_of_stop.id, 
                    trip_of_stop.destination, 
                    VehicleStop.ACT_DROPOFF, 
                    trip_of_stop.dwell_alight)
                vehicle.stop_sequence.append(vehicle_stop)

            # if len(vehicle.stop_sequence) > 0:
            #     next_stop = vehicle.stop_sequence[0]
            #     vehicle.time_at_next = time_at_next_immediate_node + NetworkHandler.travel_time(next_immediate_node,next_stop.node)
            #     next_trip = vehicle.trips[next_stop.trip_id]
            #     if next_stop.type == VehicleStop.ACT_DROPOFF and vehicle.time_at_next < next_trip.earliest_arrival_time:
            #         vehicle.time_at_next = next_trip.earliest_arrival_time

        vehicle.next_immediate_node = next_immediate_node
        vehicle.time_at_next_immediate_node = time_at_next_immediate_node
        # NOTE why do we want the last previous position as the same as the one here?
        vehicle.last_node = next_immediate_node
        vehicle.time_at_last = time_at_next_immediate_node

    @staticmethod
    def add_rebalancing_trip(vehicle: Vehicle, destination: Node, current_time: float):
        time_at_destination = current_time + NetworkHandler.travel_time(vehicle.last_node, destination)
        depot_feasible, _ = vehicle.can_return_to_depot(destination, time_at_destination)
        if depot_feasible:
        # VehicleHandler.can_return_to_depot(vehicle, destination, time_at_destination):
            vehicle.rebalancing = True
            vehicle.time_at_last = current_time
            vehicle.stop_sequence = [VehicleStop(None, destination, VehicleStop.ACT_REBALANCE, 0)]
            vehicle.time_at_next = time_at_destination
    
    @staticmethod
    def add_new_trips(vehicle: Vehicle, new_trips, prev_sequence = None, add: bool =False) -> TripInsertionPlan:
        """
        DEPRECATED but backward-compatible (split into VehicleHandler.plan_trip_insertion and Vehicle.apply)
        """
        plan = VehicleHandler.plan_trip_insertions(vehicle, new_trips, prev_sequence)
        if plan.feasible and add:
            vehicle.apply_trip_insertion(plan)
        return plan

    @staticmethod
    def plan_trip_insertions(vehicle: Vehicle, new_trips: list[Trip], prev_sequence = None) -> TripInsertionPlan:
        """
        checks feasibilty of RTV combinations (vehicle <> trips)
        Call for multiprocessing - DO NOT CHANGE shared state and must remain in VehicleHandler for that shared state, cannot be moved to the vehicle itself.
        """
        if prev_sequence is None:
            prev_sequence = []
        if vehicle.started:
            next_immediate_node, time_at_next_immediate_node = VehicleHandler.get_current_location_time(vehicle)
            trips = vehicle.trips.copy()
            # iterate trips to collect all DROPOFFs, PICKUPs, relevant nodes for travelTimeMatrix
            trips_to_pick_up = []
            trips_to_drop_off = []
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
            tt_matrix, node_indices = NetworkHandler.get_travel_time_matrix(nodes)

            # update existing sequence # NOTE why do we just overwrite the existing sequence, why can there be changes here?    
            existing_sequence = vehicle.stop_sequence
            if len(prev_sequence) > 0:
                existing_sequence = prev_sequence.copy()
            if vehicle.rebalancing:
                existing_sequence = []
            
            # calculate feasibility of trip sequence and depot return for that vehicle
            sequence, cost, sequence_feasible, last_node, time_at_last_node = VehicleHandler.get_optimal_stop_sequence(
                next_immediate_node, time_at_next_immediate_node, vehicle.am_capacity, vehicle.wc_capacity, trips, trips_to_pick_up, trips_to_drop_off, existing_sequence, tt_matrix, node_indices)
 
            if sequence_feasible: 
                # NOTE if we do not care about depot return, this entire condition should not be checked anymores
                depot_feasible, depot_travel_time = vehicle.can_return_to_depot(last_node, time_at_last_node)
                # calculate cost + travel time for later plan application
                added_cost = cost - VehicleHandler.cost_of_serving_sequence(next_immediate_node, vehicle, tt_matrix, node_indices)
                next_stop = sequence[0]
                travel_time = NetworkHandler.travel_time_from_matrix(next_immediate_node, next_stop.node, tt_matrix, node_indices) # travel to the beginning of the sequence from current position that is already planned
                console_logger.debug(f"Plan - cost: {added_cost}, final arrival: {time_at_last_node}")
                console_logger.debug("Sequence: %s", StopSequence.sequence_to_string(sequence)) 

                # TODO clean up data handover
                direct_trip_times, total_direct_travel_time, actual_travel_time, total_dwell_time, actual_route_travel_time, detour_time, idling_time = VehicleHandler._compute_trip_metrics(new_trips, sequence, next_immediate_node, time_at_next_immediate_node, tt_matrix, node_indices)

                # TODO we need a single object to handover, but these seems to be overly complex and a bad interface
                return TripInsertionPlan(
                    depot_feasible      = depot_feasible,
                    sequence_feasible   = sequence_feasible, 
                    added_cost          = added_cost,
                    sequence            = sequence,
                    trips               = new_trips,
                    next_immediate_node         = next_immediate_node,
                    time_at_next_immediate_node = time_at_next_immediate_node,
                    veh_travel_time             = travel_time,
                    depot_travel_time           = depot_travel_time,
                    direct_trip_times           = direct_trip_times,
                    total_direct_travel_time    = total_direct_travel_time,
                    actual_travel_time          = actual_travel_time,
                    total_dwell_time            = total_dwell_time,
                    actual_route_travel_time    = actual_route_travel_time,
                    detour_time                 = detour_time,
                    idling_time                 = idling_time)
            
            return TripInsertionPlan(
                    sequence_feasible= False,
                    depot_feasible = False, 
                    added_cost  = -1,
                    sequence    = None)

    @staticmethod
    def _compute_trip_metrics(
        new_trips: list[Trip],
        sequence: list[VehicleStop],
        next_immediate_node: Node,
        time_at_next_immediate_node: float,
        tt_matrix,
        node_indices
    ) -> tuple[list[float], float, float, float, float]:
        """
        Calculate key trip metrics. This avoids redundant NetworkHandler calls in the feature builder by computing everything during parallel calls and should help performance
        
        Args:
            new_trips: Trip(s) being inserted
            sequence: Feasible stop sequence after insertion
            next_immediate_node: Vehicle's current/next position
            time_at_next_immediate_node: Time at vehicle's next position
            tt_matrix: Travel time matrix
            node_indices: Node index mapping
        
        Returns:
            tuple: (direct_trip_times, total_direct_travel_time, actual_travel_time, total_dwell_time, actual_route_travel_time, detour_time, idling_time)
        """
        # Calculate sum of direct trip travel times (origin -> destination)
        # Use trip.cost which already contains the direct travel time
        direct_trip_times = []
        total_direct_travel_time = 0.0
        for trip in new_trips:
            if trip.cost is not None:
                direct_time = trip.cost
            else: # Fallback to network calculation if cost not set
                direct_time = NetworkHandler.travel_time_from_matrix(
                    trip.origin, trip.destination, tt_matrix, node_indices)
            direct_trip_times.append(direct_time)
            total_direct_travel_time += direct_time
        
        # Calculate actual route travel time by summing consecutive stops; split into travel time and dwell time
        # TODO if the first stop is only the vehicle getting to a stop while being empty, add that information separately or add it to VehicleStop, basically how much empty trip do we have?
        actual_travel_time = 0.0
        total_dwell_time = 0.0
        if sequence:
            prev_node = next_immediate_node
            for stop in sequence:
                travel_time = NetworkHandler.travel_time_from_matrix(prev_node, stop.node, tt_matrix, node_indices)
                # TODO small performance improvement: add travel_time (float) as the final value to the vehicleStop object as it is not being updated anywhere and only used here, the stops are generated in the sequence generator (had issue with pickling after adding it)
                # if stop.travel_time is not None:
                #     travel_time = stop.travel_time
                # else: # Fallback to network calculation if travel time not set
                #     travel_time = NetworkHandler.travel_time_from_matrix(prev_node, stop.node, tt_matrix, node_indices)
                #     raise ValueError("Travel time not set for stop in sequence")
                actual_travel_time += travel_time
                total_dwell_time += stop.dwell  # Accumulate dwell time separately
                prev_node = stop.node
        
        # Total route time = travel + dwell
        actual_route_travel_time = actual_travel_time + total_dwell_time
        # Detour = actual route time - sum of direct trip times
        detour_time = actual_travel_time - total_direct_travel_time
        
        # Calculate idling time (vehicle arrives before pickup is allowed)
        # NOTE not sure how useful this is currently, maybe do not use it as a feature for now
        idling_time = 0.0
        if new_trips and sequence:
            first_pickup = next((stop for stop in sequence if stop.type == 'pickup'), None)
            if first_pickup:
                time_to_reach_first_pickup = NetworkHandler.travel_time_from_matrix(
                    next_immediate_node, first_pickup.node, tt_matrix, node_indices)
                earliest_pickup_time = min(trip.pick_up_time for trip in new_trips)
                time_until_pickup_allowed = (
                    earliest_pickup_time - time_at_next_immediate_node)
                idling_time = max(0.0, time_until_pickup_allowed - time_to_reach_first_pickup)
        
        return (direct_trip_times, total_direct_travel_time, actual_travel_time, total_dwell_time, actual_route_travel_time, detour_time, idling_time)

    @staticmethod
    def cost_of_serving_sequence(next_immediate_node, vehicle, tt_matrix, node_indices):
        if vehicle.rebalancing:
            return 0
        cost = 0
        last_node = next_immediate_node
        for stop in vehicle.stop_sequence:
            cost += NetworkHandler.travel_time_from_matrix(last_node, stop.node, tt_matrix, node_indices)
            last_node = stop.node
        return cost

    @staticmethod
    def cost_of_rebalancing(vehicle, destination):
        return NetworkHandler.travel_distance(vehicle.last_node, destination)
    
    @staticmethod
    def get_optimal_stop_sequence(
            last_node, time_at_last_node, max_am_capacity, max_wc_capacity, trips, trips_to_pick_up, trips_to_drop_off, existing_sequence, tt_matrix, node_indices):
        if (len(trips_to_pick_up) + len(trips_to_drop_off)) <= VehicleHandler.LARGEST_TSP:
            return VehicleHandler.get_exact_stop_sequence(
                last_node, time_at_last_node, max_am_capacity, max_wc_capacity, trips, trips_to_pick_up, trips_to_drop_off, [], 0, tt_matrix, node_indices)
        else:
            # raise NotImplementedError("Heuristic stop sequence not implemented")
            return VehicleHandler.get_heuristic_stop_sequence(
                last_node, time_at_last_node, max_am_capacity, max_wc_capacity, trips, trips_to_pick_up, trips_to_drop_off, existing_sequence, tt_matrix, node_indices)
    
    @staticmethod
    def get_exact_stop_sequence(
            last_node, time_at_last_node, max_am_capacity, max_wc_capacity, trips, trips_to_pick_up, trips_to_drop_off, sequence, cost, tt_matrix, node_indices):   
        if len(trips_to_pick_up) == 0 and len(trips_to_drop_off) == 0:
            return sequence, cost, True, last_node, time_at_last_node
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        best_last_node, best_time_at_last_node = None, None
        # if len(trips_to_drop_off) - len(trips_to_pick_up) < max_capacity:
        for trip_id in trips_to_pick_up:
            trip = trips[trip_id]
            new_am_capacity = max_am_capacity - trip.am_capacity
            new_wc_capacity = max_wc_capacity - trip.wc_capacity
            if new_am_capacity < 0 or new_wc_capacity < 0:
                continue
            travel_time = NetworkHandler.travel_time_from_matrix(last_node,trip.origin,tt_matrix, node_indices)
            time_at_pick_up = time_at_last_node + travel_time
            if time_at_pick_up < trip.pick_up_time:
                time_at_pick_up = trip.pick_up_time

            if time_at_pick_up <= trip.latest_pick_up_time:
                time_at_pick_up = time_at_pick_up + trip.dwell_pickup
            
                new_cost = cost + NetworkHandler.travel_distance(last_node, trip.origin)
                new_trips_to_pick_up = trips_to_pick_up.copy()
                new_trips_to_pick_up.remove(trip_id)
                # TODO create the sequence at this spot instead of a list
                new_sequence = sequence.copy()
                new_sequence.append(VehicleStop(trip_id,
                                                trip.origin,
                                                VehicleStop.ACT_PICKUP,
                                                trip.dwell_pickup))
                # NOTE why do we not add the dropoff-stop here?
                # TODO why is this function calling itself
                new_sequence, new_cost, new_feasible, new_last_node, new_time_at_last_node = VehicleHandler.get_exact_stop_sequence(
                    trip.origin,
                    time_at_pick_up,
                    new_am_capacity, 
                    new_wc_capacity,
                    trips,
                    new_trips_to_pick_up,
                    trips_to_drop_off,
                    new_sequence,
                    new_cost,
                    tt_matrix, 
                    node_indices)
                if new_feasible:
                    if (not feasible) or (current_lowest_cost > new_cost):
                        current_lowest_cost = new_cost
                        feasible = new_feasible
                        best_sequence = new_sequence
                        best_last_node = new_last_node
                        best_time_at_last_node = new_time_at_last_node
        
        for trip_id in trips_to_drop_off:
            if trip_id not in trips_to_pick_up:
                trip = trips[trip_id]
                new_am_capacity = max_am_capacity + trip.am_capacity
                new_wc_capacity = max_wc_capacity + trip.wc_capacity
                travel_time = NetworkHandler.travel_time_from_matrix(last_node,trip.destination,tt_matrix, node_indices)
                time_at_drop_off = time_at_last_node + travel_time
                if time_at_drop_off <= trip.latest_arrival_time:
                    if time_at_drop_off < trip.earliest_arrival_time:
                        time_at_drop_off = trip.earliest_arrival_time
                    time_at_drop_off = time_at_drop_off + trip.dwell_alight
                    new_cost = cost + NetworkHandler.travel_distance(last_node,trip.destination)
                    new_trips_to_drop_off = trips_to_drop_off.copy()
                    new_trips_to_drop_off.remove(trip_id)
                    new_sequence = sequence.copy()
                    new_sequence.append(VehicleStop(trip_id,
                                                    trip.destination,
                                                    VehicleStop.ACT_DROPOFF,
                                                    trip.dwell_alight))
                    new_sequence, new_cost, new_feasible, new_last_node,new_time_at_last_node = VehicleHandler.get_exact_stop_sequence(trip.destination,time_at_drop_off,new_am_capacity, new_wc_capacity,trips,trips_to_pick_up,new_trips_to_drop_off,new_sequence,new_cost,tt_matrix, node_indices)
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > new_cost):
                            current_lowest_cost = new_cost
                            feasible = new_feasible
                            best_sequence = new_sequence
                            best_last_node = new_last_node
                            best_time_at_last_node = new_time_at_last_node

        return best_sequence, current_lowest_cost, feasible, best_last_node, best_time_at_last_node

    @staticmethod
    def get_heuristic_stop_sequence(
            last_node, time_at_last_node, max_am_capacity, max_wc_capacity, trips, trips_to_pick_up, trips_to_drop_off, existing_sequence, tt_matrix, node_indices):
        feasible = False
        best_sequence = None
        current_lowest_cost = -1
        current_am_capacity, current_wc_capacity = max_am_capacity,max_wc_capacity
        best_last_node, best_time_at_last_node = None, None

        trips_not_in_sequence = set(trips_to_pick_up)
        for stop in existing_sequence:
            if stop.trip_id in trips_not_in_sequence:
                trips_not_in_sequence.remove(stop.trip_id)
        if len(trips_not_in_sequence) == 0:
            new_feasible, cost, current_last_node, current_last_time  = VehicleHandler.evaluate_stop_sequence(
                last_node, time_at_last_node, trips, existing_sequence, max_am_capacity, max_wc_capacity, tt_matrix, node_indices)
            return existing_sequence, cost, True, current_last_node, current_last_time

        for trip_id in trips_not_in_sequence:
            new_trip = trips[trip_id]
            feasible = False
            best_sequence = None
            current_lowest_cost = -1
            for pick_up_index in range(len(existing_sequence)+1):
                for drop_off_index in range(pick_up_index+1,len(existing_sequence)+2):
                    new_sequence = existing_sequence.copy()
                    trip_id = new_trip.id
                    new_sequence.insert(pick_up_index,VehicleStop(trip_id,
                                                                  new_trip.origin,
                                                                  VehicleStop.ACT_PICKUP,
                                                                  new_trip.dwell_pickup))
                    new_sequence.insert(drop_off_index,VehicleStop(trip_id,
                                                                   new_trip.destination,
                                                                   VehicleStop.ACT_DROPOFF,
                                                                   new_trip.dwell_alight))
                    new_feasible, cost, current_last_node, current_last_time  = VehicleHandler.evaluate_stop_sequence(
                        last_node,
                        time_at_last_node,
                        trips,new_sequence, 
                        max_am_capacity, 
                        max_wc_capacity, 
                        tt_matrix, 
                        node_indices)
                    if new_feasible:
                        if (not feasible) or (current_lowest_cost > cost):
                            current_lowest_cost = cost
                            feasible = new_feasible
                            best_sequence = new_sequence
                            best_last_node = current_last_node
                            best_time_at_last_node = current_last_time
            if feasible:
                existing_sequence = best_sequence
            else:
                break
        return best_sequence, current_lowest_cost, feasible, best_last_node, best_time_at_last_node

    @staticmethod
    def evaluate_stop_sequence(last_node, time_at_last_node, trips, new_sequence, max_am_capacity, max_wc_capacity, tt_matrix, node_indices):
        # TODO check validity of stops and PICKUP + DROPOFF guarantee
        cost = 0
        new_feasible = True
        current_time = time_at_last_node
        current_node = last_node
        current_am_capacity, current_wc_capacity = max_am_capacity, max_wc_capacity
        for stop in new_sequence:
            trip = trips[stop.trip_id]
            travel_time = NetworkHandler.travel_time_from_matrix(current_node, stop.node, tt_matrix, node_indices)
            cost +=  NetworkHandler.travel_distance(current_node,stop.node)
            current_time = current_time + travel_time
            if stop.type == VehicleStop.ACT_PICKUP:
                if current_time < trip.pick_up_time:
                    current_time = trip.pick_up_time
                current_am_capacity -= trip.am_capacity
                current_wc_capacity -= trip.wc_capacity
                if min(current_am_capacity,current_wc_capacity) < 0 or current_time > trip.latest_pick_up_time:
                    new_feasible = False
                    break
                current_time = current_time + trip.dwell_pickup
            else:
                current_am_capacity += trip.am_capacity
                current_wc_capacity += trip.wc_capacity
                if current_time > trip.latest_arrival_time:
                    new_feasible = False
                    break
                if current_time < trip.earliest_arrival_time:
                    current_time = trip.earliest_arrival_time
                current_time = current_time + trip.dwell_alight
            current_node = stop.node
        return new_feasible, cost, current_node, current_time

    @staticmethod
    def can_serve_trips(trips, new_trip, current_sequence):
        """TODO add docstring"""
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
        # if len(current_sequence) == 0:
        current_time = new_trip.pick_up_time
        for trip_id in trips:
            if trips[trip_id].pick_up_time < current_time:
                current_time = trips[trip_id].pick_up_time
            starting_locations.append(trips[trip_id].origin)
        # else:
        #     starting_locations.append(current_sequence[0].node)
        #     starting_locations.append(trips[new_trip].origin)
        for starting_location in starting_locations:
            sequence, cost, t_feasible = None, None, None
            if 2 * len(trips) <= VehicleHandler.LARGEST_TSP:
                sequence, cost, t_feasible, last_node, time_at_last_node = VehicleHandler.get_exact_stop_sequence(
                    starting_location,
                    current_time,
                    VehicleHandler.MAX_AM_CAPACITY,
                    VehicleHandler.MAX_WC_CAPACITY,
                    trips,
                    trips_to_pick_up,
                    trips_to_drop_off,
                    [],
                    0,
                    tt_matrix, 
                    node_indices)
            else:
                sequence, cost, t_feasible, last_node, time_at_last_node = VehicleHandler.get_heuristic_stop_sequence(
                    starting_location,
                    current_time,
                    VehicleHandler.MAX_AM_CAPACITY,
                    VehicleHandler.MAX_WC_CAPACITY,
                    trips,
                    [new_trip.id],
                    [new_trip.id],
                    current_sequence,
                    tt_matrix,
                    node_indices)
            if t_feasible:
                if not feasible or best_cost > cost:
                    feasible = t_feasible
                    best_cost = cost
                    best_sequence = sequence
        return feasible, best_cost, best_sequence

    # BELOW DEPRECATED METHODS - ONLY USED IN SIMULATION APPROACH
    def simulate_vehicle(self, vehicle, current_time):
        """
        TODO check if this doc comment is correct or whether it needs an update
        TODO certain steps are never reached
        TODO logging never shows a pickup and uncertain why it never gets there

        Simulate vehicle state forward to a given current_time and apply any stop, pickup/dropoff,
        dwelling, rebalancing and routing decisions that become active at or before that time.
        Parameters
        ----------
        current_time : numeric
            The simulation time instant to advance the vehicle to.
        vehicle : object
            Vehicle instance whose mutable state is advanced. Expected attributes used by this
            routine include: id, start_time, started, rebalancing, time_at_next, stop_sequence,
            last_node, time_at_last, final_stop_time, dwelling, time_at_next_immediate_node,
            next_immediate_node, picked, am_capacity, wc_capacity, trips (dict), picked (list/set),
            etc. Stop objects in stop_sequence are expected to have: node, trip_id, type, dwell
            and fields written here: stop_time, vehicle_id, request_id. Trip objects are expected
            to provide: request_id, am_capacity, wc_capacity, pick_up_time, earliest_arrival_time,
            picked boolean.
        Returns
        -------
        tuple (completed_stops, picked_requests, completed_requests)
            completed_stops : list
                Stops that were reached at or before current_time and were logged by this call.
            picked_requests : list
                Request IDs that were picked up during this call.
            completed_requests : list
                Request IDs that were dropped off (completed) during this call.
        Behavior / Simulation flow and decision order
        --------------------------------------------
        1. Start check
           - If current_time >= vehicle.start_time and vehicle.started is False, set vehicle.started = True.
        2. Rebalancing arrival (highest priority)
           - If vehicle.rebalancing is True and current_time >= vehicle.time_at_next:
             - Pop the next stop from stop_sequence, mark it as completed (set stop_time, vehicle_id),
               update vehicle.last_node, vehicle.time_at_last and vehicle.final_stop_time,
               clear rebalancing flag and append the stop to completed_stops.
             - Return immediately with the single completed stop and empty picked/completed lists.
             - (Note: rebalancing arrival is treated as an atomic arrival and the function exits early.)
        3. End dwelling
           - If vehicle.dwelling is True and vehicle.time_at_last <= current_time, clear dwelling flag.
           - If not dwelling and stop_sequence is empty, update time_at_last, time_at_next_immediate_node
             and next_immediate_node to reflect the vehicle being idle at its last_node at current_time.
        4. Process scheduled stops that are due (loop)
           - While there are stops in stop_sequence and current_time >= vehicle.time_at_next:
             - Pop the next stop and log it (set stop_time, trip.request_id, vehicle_id) and append to completed_stops.
             - Update vehicle.last_node to the stop node and vehicle.time_at_last to time_at_next + stop.dwell.
             - Set final_stop_time to time_at_last.
             - If time_at_last > current_time, set dwelling=True (vehicle is dwelling through current_time).
             - If stop is a PICKUP:
                 - Add trip_id to vehicle.picked, decrement vehicle capacities by the trip capacity,
                   mark trip.picked = True and append the trip's request_id to picked_requests.
               If stop is a DROP_OFF:
                 - Remove trip_id from vehicle.picked, increment vehicle capacities back,
                   append the trip's request_id to completed_requests and delete the trip from vehicle.trips.
             - If there are more stops left in the sequence, compute vehicle.time_at_next as:
                 time_at_last + NetworkHandler.travel_time(last_node, next_stop.node)
               and then enforce time-window constraints:
                 - For PICKUP: if time_at_next < trip.pick_up_time, set time_at_next = trip.pick_up_time
                 - For DROP_OFF: if time_at_next < trip.earliest_arrival_time, set time_at_next = earliest_arrival_time
             - The loop continues to consume any additional stops whose time_at_next <= current_time.
        5. Update in-transit immediate node (when not all stops are completed)
           - If stop_sequence is non-empty after processing arrivals:
             - Compute ori = vehicle.last_node and dest = stop_sequence[0].node.
             - Default next_immediate_node and time_at_next_immediate_node are last_node and time_at_last.
             - If not dwelling: call NetworkHandler.get_current_location_time(ori, dest, vehicle.time_at_last, current_time)
               to obtain the current location along the link (node/time) at current_time and update
               next_immediate_node and time_at_next_immediate_node accordingly.
             - Assign these values into vehicle.time_at_next_immediate_node and vehicle.next_immediate_node,
               and also update vehicle.last_node and vehicle.time_at_last to reflect the vehicle’s in-transit
               location at current_time.
        6. Recompute stop sequence for non-rebalancing vehicles (pruning & replanning)
           - If the vehicle is not rebalancing, prune vehicle.trips to only keep trips that are currently picked.
             This ensures future decisions are only about drop-offs for onboard passengers.
           - Build a new stop_sequence comprised of the DROP_OFF stops corresponding to the picked trips,
             preserving the node list starting with vehicle.next_immediate_node.
           - If there are any picked trips:
             - Query NetworkHandler.get_travel_time_matrix(nodes) for pairwise travel times between nodes.
             - Call VehicleHandler.get_exact_stop_sequence(...) to compute an optimized feasible drop-off sequence
               starting from next_immediate_node, with the vehicle's current capacities and updated trips.
             - If the returned plan is feasible, replace vehicle.stop_sequence with the best_sequence.
             - Set vehicle.time_at_next to time_at_next_immediate_node + travel_time to the first stop in the new sequence,
               and enforce any earliest_arrival_time constraints for the first drop-off as in step 4.
        7. Exit and return
           - Return the lists: completed_stops, picked_requests, completed_requests.
        Side effects and important notes
        -------------------------------
        - This function mutates the vehicle object heavily (stop_sequence, trips, picked, capacities,
          last_node, time_at_last, time_at_next, dwelling, rebalancing, next_immediate_node, etc.).
        - It logs completed stops by setting fields on stop objects (stop_time, vehicle_id, request_id).
        - It handles one rebalancing arrival as a special case and returns immediately after executing it.
        - Time-window constraints (pick_up_time, earliest_arrival_time) are always enforced when scheduling
          the next arrival time (time_at_next).
        - External dependencies: NetworkHandler.travel_time, NetworkHandler.get_current_location_time,
          NetworkHandler.get_travel_time_matrix, and VehicleHandler.get_exact_stop_sequence.
        - Assumes constants VehicleStop.ACT_PICKUP and VehicleStop.ACT_DROPOFF are defined and used to distinguish stop types.
        """
        # track state changes
        completed_stops = []
        picked_requests = []
        completed_requests = []
        
        # start vehicles that are active now
        if current_time >= vehicle.start_time:
            if not vehicle.started:
                vehicle.started = True
        
        # Handle whether vehicle has been rebalancing (atomic arrival exits simulation early)
        if vehicle.rebalancing and current_time >= vehicle.time_at_next:
            next_stop = vehicle.stop_sequence.pop(0)
            vehicle.last_node = next_stop.node
            vehicle.time_at_last = vehicle.time_at_next
            vehicle.final_stop_time = vehicle.time_at_last
            vehicle.rebalancing = False
            # logging the stop
            next_stop.stop_time = vehicle.time_at_next
            next_stop.vehicle_id = vehicle.id
            completed_stops.append(next_stop)
            return completed_stops, picked_requests, completed_requests
        
        # Handle dwelling and turn vehicle to idle if no stops are left at the current location
        if vehicle.dwelling and vehicle.time_at_last <= current_time:
            vehicle.dwelling = False
        
        if (not vehicle.dwelling) and len(vehicle.stop_sequence) == 0:
            vehicle.time_at_last = current_time
            vehicle.time_at_next_immediate_node = current_time
            vehicle.next_immediate_node = vehicle.last_node

        # Process complete vehicle stops until current time to fix all prior decisions
        while len(vehicle.stop_sequence) > 0 and current_time >= vehicle.time_at_next:
            next_stop = vehicle.stop_sequence.pop(0)
            # logging the stop
            trip = vehicle.trips[next_stop.trip_id]
            next_stop.stop_time = vehicle.time_at_next
            
            next_stop.request_id = trip.request_id
            next_stop.vehicle_id = vehicle.id
            completed_stops.append(next_stop)

            vehicle.last_node = next_stop.node
            vehicle.time_at_last = vehicle.time_at_next + next_stop.dwell
            vehicle.final_stop_time = vehicle.time_at_last
            if vehicle.time_at_last > current_time:
                vehicle.dwelling = True
            if next_stop.type == VehicleStop.ACT_PICKUP:
                vehicle.picked.append(next_stop.trip_id)
                vehicle.am_capacity = vehicle.am_capacity - trip.am_capacity
                vehicle.wc_capacity = vehicle.wc_capacity - trip.wc_capacity
                picked_trip = vehicle.trips[next_stop.trip_id]
                picked_trip.picked = True
                picked_requests.append(picked_trip.request_id)
                console_logger.info(f"pickup: {next_stop.trip_id}") #, picked_trip.request_id)
            else: # dropoff request
                vehicle.picked.remove(next_stop.trip_id)
                completed_requests.append(vehicle.trips[next_stop.trip_id].request_id)
                vehicle.am_capacity = vehicle.am_capacity + trip.am_capacity
                vehicle.wc_capacity = vehicle.wc_capacity + trip.wc_capacity

                console_logger.info(f"dropoff: {next_stop.trip_id}")
                del vehicle.trips[next_stop.trip_id]
            if len(vehicle.stop_sequence) > 0:
                next_stop = vehicle.stop_sequence[0]
                vehicle.time_at_next = vehicle.time_at_last + NetworkHandler.travel_time(vehicle.last_node,next_stop.node)
                next_trip = vehicle.trips[next_stop.trip_id]
                if next_stop.type == VehicleStop.ACT_PICKUP and vehicle.time_at_next < next_trip.pick_up_time:
                    vehicle.time_at_next = next_trip.pick_up_time
                if next_stop.type == VehicleStop.ACT_DROPOFF and vehicle.time_at_next < next_trip.earliest_arrival_time:
                    vehicle.time_at_next = next_trip.earliest_arrival_time
        
        # update in-transit vehicle and trip states
        if len(vehicle.stop_sequence) > 0:
            ori =  vehicle.last_node
            dest = vehicle.stop_sequence[0].node
            next_immediate_node= vehicle.last_node
            time_at_next_immediate_node = vehicle.time_at_last
            if not vehicle.dwelling:
                time_at_next_immediate_node, next_immediate_node = NetworkHandler.get_current_location_time(ori, dest, vehicle.time_at_last, current_time)
            vehicle.time_at_next_immediate_node = time_at_next_immediate_node
            vehicle.next_immediate_node = next_immediate_node
            vehicle.last_node,vehicle.time_at_last = next_immediate_node, time_at_next_immediate_node
            
            if not vehicle.rebalancing:
                updated_trip_list = {}
                trips_to_drop_off = []
                for trip_id in vehicle.trips:
                    if trip_id in vehicle.picked:
                        # TODO why is this step never reached?
                        console_logger.info(f"To be dropped off: v{vehicle.id}, t{trip_id}")
                        updated_trip_list[trip_id] = vehicle.trips[trip_id]
                        trips_to_drop_off.append(trip_id)
                
                vehicle.trips = updated_trip_list
                existing_sequence = []
                nodes = [vehicle.next_immediate_node]
                for stop in vehicle.stop_sequence:
                    if stop.trip_id in updated_trip_list and stop.type == VehicleStop.ACT_DROPOFF:
                        existing_sequence.append(stop)
                        nodes.append(stop.node)
                vehicle.stop_sequence = existing_sequence
                
                # TODO what does this do? is this on the right level or where should this be called?
                if len(vehicle.picked) > 0:
                    tt_matrix, node_indices = NetworkHandler.get_travel_time_matrix(nodes)
                    # TODO: JW cost (third to last element added as 0 in order to get the code running)
                    best_sequence, _, feasible, _, _ = VehicleHandler.get_exact_stop_sequence(next_immediate_node,time_at_next_immediate_node,vehicle.am_capacity,vehicle.wc_capacity,updated_trip_list,[],trips_to_drop_off,[],0, tt_matrix, node_indices)
                    
                    if feasible:
                        console_logger.debug(f"Vehicle {vehicle.id}, new sequence: {[ (stop.trip_id, stop.type) for stop in vehicle.stop_sequence]}")
                        vehicle.stop_sequence = best_sequence
                    next_stop = vehicle.stop_sequence[0]
                    vehicle.time_at_next = time_at_next_immediate_node + NetworkHandler.travel_time(next_immediate_node,next_stop.node)
                    next_trip = vehicle.trips[next_stop.trip_id]
                    if next_stop.type == VehicleStop.ACT_DROPOFF and vehicle.time_at_next < next_trip.earliest_arrival_time:
                        vehicle.time_at_next = next_trip.earliest_arrival_time
        
        return completed_stops, picked_requests, completed_requests

    def simulate_vehicles(self, current_time):
        """ iterate over each vehicle and collect all the updated information what happened in the last step"""
        completed_stops = []
        picked_requests = []
        completed_requests = []
        completed_vehicles = []

        for vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            if vehicle.has_completed_operations(current_time):
                stop = vehicle.return_to_depot()
                completed_stops.append(stop)
                completed_vehicles.append(vehicle_id)
            else:
                veh_completed_stops, veh_picked_requests, veh_completed_requests = self.simulate_vehicle(vehicle, current_time)
                completed_stops.extend(veh_completed_stops)
                picked_requests.extend(veh_picked_requests)
                completed_requests.extend(veh_completed_requests)
        
        for vehicle_id in completed_vehicles:
            del self.vehicles[vehicle_id]
        return completed_stops, picked_requests, completed_requests

    def get_state(self, driver_run):
        new_state = driver_run[PayloadParser.DRIVER_STATE]
        current_order = new_state[PayloadParser.DRIVER_STATE_LOC_SERV]
        manifest = driver_run[PayloadParser.DRIVER_MANIFEST][:current_order]
        vehicle = self.vehicles[new_state[PayloadParser.DRIVER_STATE_RUN_ID]]
        next_immediate_node,time_at_next_immediate_node = VehicleHandler.get_current_location_time(vehicle)
        new_state[PayloadParser.DRIVER_STATE_LOC] = {"lat": next_immediate_node.lat,
                                                     "lon": next_immediate_node.lon}
        new_state[PayloadParser.DRIVER_STATE_DT_SEC] = time_at_next_immediate_node
        manifest.extend(VehicleHandler.get_manifest(vehicle, current_order))
        new_driver_run = {PayloadParser.DRIVER_STATE:new_state,PayloadParser.DRIVER_MANIFEST:manifest}
        return new_driver_run

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

    # BELOW UNUSED METHODS
    def save_snapshot(self):
        """deprecated"""
        with open(self.config.OUTPUT_DIR + "vehicle_snapshot.p", 'wb') as snapshot_file:
            pickle.dump(self, snapshot_file)

    def load_snapshot(snapshot_directory):
        """deprecated"""
        snapshot = None
        with open(snapshot_directory+"vehicle_snapshot.p", 'rb') as snapshot_file:
            snapshot = pickle.load(snapshot_file)
        return snapshot
    
    def read_vehicles(self, filename, starting_date, max_number_of_vehicles):
        """deprecated"""
        # strings only relevant for one specific function
        START_TIME = 'start_time'
        CAPACITY = 'capacity'
        ID = 'id'
        dateparse = lambda x: datetime.strptime(x, '%H:%M:%S')
        data = pd.read_csv(
            filename,
            parse_dates=[START_TIME],
            date_parser=dateparse).sort_values(by = [START_TIME])
        for _, row in data.iterrows():
            self.count+=1
            capacity = min(int(row[CAPACITY]), self.MAX_CAPACITY)
            id = int(row[ID])
            start_time = starting_date + timedelta(hours=row[START_TIME].hour,
                                                   minutes=row[START_TIME].minute,seconds=row[START_TIME].second)
            nearest_lat,nearest_lon = NetworkHandler.get_nearest_node(float(row.lat),float(row.lon))
            vehicle = Vehicle(id,
                              Node(nearest_lat,nearest_lon), 
                              capacity, 
                              start_time)
            self.vehicles[id] = vehicle
            self.MAX_CAPACITY = max(capacity, self.MAX_CAPACITY)
            if self.count == max_number_of_vehicles:
                break