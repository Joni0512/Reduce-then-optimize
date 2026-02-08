import logging

from dataclasses import dataclass

from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.node import Node

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser

@dataclass
class TripInsertionPlan:
    depot_feasible: bool = False
    sequence_feasible: bool = False
    # feasible: bool = False# sequence_feasible && depot_feasible
    added_cost: float = -1
    sequence: list[VehicleStop] = None
    trips: list[Trip] = None # TODO turn into a dict[id, Trip] where I can more easily access the information, it should also be clearly limited to the scope of tripAssignment as far as I can tell
    next_immediate_node: Node = None
    time_at_next_immediate_node: float = None
    # add information on veh_travel from last vehicle location to next_immediate_node, and depot_return
    veh_travel_time: float = None
    depot_travel_time: float = None
    
class Vehicle:
    """Vehicle-related information covering basic vehicle information and simulation state during runtime"""
    def __init__(self, 
                 vehicle_id: int, 
                 start_node: Node, 
                 am_capacity: int, 
                 wc_capacity: int, 
                 start_time: float, 
                 end_time: float, 
                 depot: Node):
        # Static vehicle information
        self.id = vehicle_id
        self.start_time = start_time
        self.end_time = end_time
        self.depot = depot
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
    
        # tracks simulation state
        # TODO add bool for inactive vehicles that are not in operation anymore (either time ran out or returned to depot)
        self.started = False
        self.rebalancing = False
        self.dwelling = False
        self.time_at_last = start_time
        self.last_node = start_node
        self.time_at_next = start_time
        self.next_immediate_node = start_node
        self.trips: dict[int, Trip] = {}
        self.picked = []
        self.served_trips = []
        self.stop_sequence: list[VehicleStop] = [] # TODO add StopSequence wherever this list is updated or used
        self.final_stop_time = start_time # TODO this is never updated

    def get_current_location_time(self):
        # TODO bad interface as it requires a separate previous update of self.time_at_next_immediate_node, self.next_immediate_node to get the correct data in this moment OR an update of the position in the stop_sequence in this moment
        next_immediate_node = self.last_node
        time_at_next_immediate_node = self.time_at_last
        if len(self.stop_sequence) > 0:
            time_at_next_immediate_node = self.time_at_next_immediate_node
            next_immediate_node = self.next_immediate_node
        return next_immediate_node, time_at_next_immediate_node
    
    def can_return_to_depot(self, last_node: Node, time_at_last_node: int) -> tuple[bool, float]:
        """
        checks if the vehicle can still reach the depot after fulfilling its last service, also calculates the travel_time directly in order to facilitate future application
        
        :return: depot_feasible: checks whether depot can still be reached in-time
        :rtype: bool
        """
        travel_time = NetworkHandler.travel_time(last_node, self.depot)
        if time_at_last_node + travel_time < self.end_time:
            return True, travel_time
        return False, -1
    
    # BELOW DEPRECATED METHODS FOR SIMULATION
    def has_completed_operations(self, current_time: int):
        """boolean for checking whether the vehicle's operational time has run out and the vehicle is empty."""
        if len(self.stop_sequence) == 0 and self.end_time <= current_time:
            assert len(self.picked) == 0
            return True
        return False

    def return_to_depot(self) -> VehicleStop:
        """handle steps to build stop to return to depot."""
        # TODO FIXME this currently does not work, the final_stop_time does not seem to be correct as it is never updated
        stop = VehicleStop(None, self.depot, VehicleStop.ACT_DEPOT, 0)
        stop.stop_time = self.final_stop_time + NetworkHandler.travel_time(self.last_node, self.depot)
        stop.vehicle_id = self.id 
        return stop
    
    def restore_state_from_manifest(self, driver_run, boarded_requests, boarded_trips, dwell_alight, dwell_pickup):
        """
        @deprecated or at least not used anymore
        
        restore current state of the vehicle based on the current boarded requests and the already finished manifest
        """        
        state = driver_run[PayloadParser.DRIVER_STATE]
        current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        self.started = True
        time_at_next_immediate_node = state[PayloadParser.DRIVER_STATE_DT_SEC]
        location = state[PayloadParser.DRIVER_STATE_LOC]
        next_immediate_node = NetworkHandler.get_node_from_manifest_location(
            location, 
            node_id = NetworkHandler.get_next_node_id(location['lat'], location['lon']))

        manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
        if len(manifest) > 0:
            for stop in manifest:
                if stop[PayloadParser.MANIFEST_ORDER] > current_order:
                    break
                if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                    self.am_capacity -= stop[PayloadParser.MANIFEST_AMBULATORY]
                    self.wc_capacity -= stop[PayloadParser.MANIFEST_WHEELCHAIR]
                else: # dropoff
                    self.am_capacity += stop[PayloadParser.MANIFEST_AMBULATORY]
                    self.wc_capacity += stop[PayloadParser.MANIFEST_WHEELCHAIR]
            
            # Adding existing route to the vehicle
            filtered_manifest = []
            for stop in manifest:
                booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                if booking_id in boarded_requests and stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_DROPOFF:
                    filtered_manifest.append(stop)

            for stop in filtered_manifest:
                trip_of_stop = None
                for trip in boarded_trips:
                    if trip.request_id == stop[PayloadParser.MANIFEST_BOOKING_ID]:
                        trip_of_stop = trip
                        break
                self.trips[trip_of_stop.id] = trip_of_stop
                
                self.picked.append(trip_of_stop.id)
                vehicle_stop = VehicleStop(trip_of_stop.id, 
                                           trip_of_stop.destination, 
                                           VehicleStop.ACT_DROPOFF,
                                           trip_of_stop.dwell_alight)
                self.stop_sequence.append(vehicle_stop)

            # if len(vehicle.stop_sequence) > 0:
            #     next_stop = vehicle.stop_sequence[0]
            #     vehicle.time_at_next = time_at_next_immediate_node + NetworkHandler.travel_time(next_immediate_node,next_stop.node)
            #     next_trip = vehicle.trips[next_stop.trip_id]
            #     if next_stop.type == VehicleStop.ACT_DROPOFF and vehicle.time_at_next < next_trip.earliest_arrival_time:
            #         vehicle.time_at_next = next_trip.earliest_arrival_time

        self.next_immediate_node = next_immediate_node
        self.time_at_next_immediate_node = time_at_next_immediate_node
        self.last_node = next_immediate_node
        self.time_at_last = time_at_next_immediate_node

    def apply_trip_insertion(self, plan: TripInsertionPlan):    
        """
        Updates vehicle state based on the trip plan that is be applied

        :param TripInsertionPlan plan: stores trip details for application
        """
        # read and apply relevant information from plan to vehicle 
        self.rebalancing = False
        self.last_node = plan.next_immediate_node
        self.time_at_last = plan.time_at_next_immediate_node
        self.time_at_next = self.time_at_last + plan.veh_travel_time
        for trip in plan.trips:
            self.trips[trip.id] = trip
        self.stop_sequence = plan.sequence

        # update vehicle state
        next_stop = self.stop_sequence[0]
        next_trip = self.trips[next_stop.trip_id]
        if next_stop.type == VehicleStop.ACT_PICKUP and self.time_at_next < next_trip.pick_up_time:
            self.time_at_next = next_trip.pick_up_time
        if next_stop.type == VehicleStop.ACT_DROPOFF and self.time_at_next < next_trip.earliest_arrival_time:
            self.time_at_next = next_trip.earliest_arrival_time

    def __str__(self):
        trip_str = "{" + ', '.join([f"{trip_id}: {str(trip)}" for trip_id, trip in self.trips.items()]) + "}"
        picked_str = ', '.join([str(pick) for pick in self.picked])
        served_str = ', '.join([str(served) for served in self.served_trips])
        return f"Vehicle ID {self.id}: time: {self.start_time}>{self.end_time}, trips: {trip_str}, picked: [{picked_str}], served: [{served_str}], last_node: {self.last_node}, stop_sequence: [{self.stop_sequence}]"