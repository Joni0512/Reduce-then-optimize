from enum import Enum
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser # TODO move the definition of the dict-keys (strings to a different place) or rather the interfaces should not be required here anymore
from rtv_solver.structure.sequence import StopSequence

class VehicleStatus(Enum):
    # TODO add status instead of deleting vehicles when they are not active anymore; with an enum makes it easier
    ACTIVE = 1 # during operation
    INACTIVE = 2 # before it is active while in depot
    COMPLETED = 3 # after end of operations
    
class Vehicle:
    """Vehicle-related information covering basic vehicle information and simulation state during runtime"""
    # TODO differentiate initialized information and dynamic information to keep track of state
    def __init__(self, 
                 id, 
                 start_node, 
                 am_capacity, 
                 wc_capacity, 
                 start_time, 
                 end_time, 
                 depot):
        # Static vehicle information
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.depot = depot
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
        
        # tracks simulation state
        # TODO move simulation step into this class as it is based on the vehicle, simulation step not used anymore
        self.started = False
        self.rebalancing = False
        self.dwelling = False
        self.time_at_last = start_time
        self.time_at_next = start_time
        self.trips = {}
        self.picked = []
        self.served_trips = []
        self.last_node = start_node
        self.stop_sequence: list[VehicleStop] = []
        self.final_stop_time = start_time

    def set_sequence(self, new_sequence: list[VehicleStop]):
        """method should update the sequence if stops are added but not reset how it is currently build
        NOTE rebuild in a way that gives us future control how the updates are made but still easier to debug"""
        self.stop_sequence = StopSequence(new_sequence)
        # TODO how code should work in order to manage the car properly, it should rather append stops and for dropoffs it should check whether it has previously been picked up  

    def get_current_location_time(self):
        next_immediate_node = self.last_node
        time_at_next_immediate_node = self.time_at_last
        if len(self.stop_sequence)>0:
            time_at_next_immediate_node = self.time_at_next_immediate_node
            next_immediate_node = self.next_immediate_node
        return next_immediate_node,time_at_next_immediate_node
    
    def has_completed_operations(self, current_time: int):
        """boolean for checking whether the vehicle's operational time has run out and the vehicle is empty."""
        if len(self.stop_sequence) == 0 and self.end_time <= current_time:
            assert len(self.picked) == 0 # TODO currently code fails on this obvious condition
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
        next_immediate_node = NetworkHandler.manifest_location(
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

    def __str__(self):
        trip_str = "{" + ', '.join([f"{trip_id}: {str(trip)}" for trip_id, trip in self.trips.items()]) + "}"
        picked_str = ', '.join([str(pick) for pick in self.picked])
        served_str = ', '.join([str(served) for served in self.served_trips])
        return f"Vehicle ID {self.id}: time: {self.start_time}>{self.end_time}, trips: {trip_str}, picked: [{picked_str}], served: [{served_str}], last_node: {self.last_node}, stop_sequence: [{self.stop_sequence}]"