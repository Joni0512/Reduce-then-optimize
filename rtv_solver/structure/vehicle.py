from enum import Enum
from dataclasses import dataclass
from typing import Optional

from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.sequence import StopSequence
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.node import Node

@dataclass
class TripInsertionPlan:
    feasible: bool
    added_cost: float
    sequence: list[VehicleStop]
    trips: Optional[list] = None
    next_immediate_node: Optional[Node] = None
    time_at_next_immediate_node: Optional[float] = None
    veh_travel_time: Optional[float] = None
    
class Vehicle:
    """Vehicle-related information covering basic vehicle information and simulation state during runtime"""
    def __init__(self, 
                 vehicle_id, 
                 start_node, 
                 am_capacity, 
                 wc_capacity, 
                 start_time, 
                 end_time, 
                 depot):
        # Static vehicle information
        self.id = vehicle_id
        self.start_time = start_time
        self.end_time = end_time
        self.depot = depot
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
    
        # tracks simulation state
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
        self.final_stop_time = start_time 

    def get_current_location_time(self):
        next_immediate_node = self.last_node
        time_at_next_immediate_node = self.time_at_last
        if len(self.stop_sequence)>0:
            time_at_next_immediate_node = self.time_at_next_immediate_node
            next_immediate_node = self.next_immediate_node
        return next_immediate_node,time_at_next_immediate_node
    
    def has_completed_operations(self, current_time: int):
        # TODO delete when not used
        """boolean for checking whether the vehicle's operational time has run out and the vehicle is empty."""
        if len(self.stop_sequence) == 0 and self.end_time <= current_time:
            assert len(self.picked) == 0
            # problem is not located here as the self.picked needs to be offloaded somehow
            print(f"Vehicle {self.id} completed its operation and returned to depot.")
            return True
        else:
            return False

    def return_to_depot(self):
        """handle steps to build stop to return to depot."""
        stop = VehicleStop(None, self.depot, VehicleStop.ACT_DEPOT, 0)
        stop.stop_time = self.final_stop_time + NetworkHandler.travel_time(self.last_node,self.depot)
        stop.vehicle_id = self.id    
        return stop
    
    # only used in Online_RTV_solver
    def restore_state_from_manifest(self, driver_run, boarded_requests, boarded_trips, dwell_alight, dwell_pickup):
        """restore current state of the vehicle based on the current boarded requests and the already finished manifest"""        
        state = driver_run[PayloadParser.DRIVER_STATE]
        current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
        self.started = True
        time_at_next_immediate_node = state[PayloadParser.DRIVER_STATE_DT_SEC]
        location = state[PayloadParser.DRIVER_STATE_LOC]
        next_immediate_node = NetworkHandler.get_node_from_manifest_location(
            location, 
            node_id = NetworkHandler.get_next_node_id(location['lat'], location['lon']))

        manifest = driver_run["manifest"]
        if len(manifest) > 0:
            for stop in manifest:
                if stop["order"] > current_order:
                    break
                if stop["action"] == VehicleStop.ACT_PICKUP:
                    self.am_capacity -= stop["am"]
                    self.wc_capacity -= stop["wc"]
                else: # dropoff
                    self.am_capacity += stop["am"]
                    self.wc_capacity += stop["wc"]
            
            # Adding existing route to the vehicle
            filtered_manifest = []
            for stop in manifest:
                booking_id = stop['booking_id']
                if booking_id in boarded_requests and stop['action'] == VehicleStop.ACT_DROPOFF:
                    filtered_manifest.append(stop)

            for stop in filtered_manifest:
                trip_of_stop = None
                for trip in boarded_trips:
                    if trip.request_id == stop['booking_id']:
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
        Adds the trip stored in the TripInsertionPlan as it was all previously checked
        """
        # read relevant information from plan
        trips = plan.trips
        sequence = plan.sequence
        next_immediate_node = plan.next_immediate_node
        time_at_next_immediate_node = plan.time_at_next_immediate_node
        travel_time = plan.veh_travel_time
        # update vehicle 
        self.rebalancing = False
        self.last_node = next_immediate_node
        self.time_at_last = time_at_next_immediate_node
        self.time_at_next = self.time_at_last + travel_time
        for trip in trips:
            self.trips[trip.id] = trip
        
        self.stop_sequence = sequence
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