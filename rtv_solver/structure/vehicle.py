import logging

from dataclasses import dataclass

from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.node import Node
from rtv_solver.structure.driver_run import ManifestEntry

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser

# TODO move to a separate file in order to keep it more readable
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

    # preprocessed trip metrics in order to build features for each trip-vehicle combinations
    direct_trip_times: list[float] = None
    total_direct_travel_time: float = None
    actual_travel_time: float = None
    total_dwell_time: float = None
    actual_route_travel_time: float = None
    detour_time: float = None
    idling_time: float = None
    
class Vehicle:
    """Vehicle-related information covering basic vehicle information and simulation state during runtime"""
    def __init__(self, 
                 vehicle_id: int, 
                 start_node: Node, 
                 am_capacity: int, 
                 wc_capacity: int, 
                 start_time: float, 
                 end_time: float, 
                 depot: Node,
                 manifest: list[ManifestEntry]):
        # Static vehicle information
        self.id = vehicle_id
        self.start_time = start_time
        self.end_time = end_time
        self.depot = depot
        self.am_capacity = am_capacity
        self.wc_capacity = wc_capacity
    
        # tracks simulation state
        # TODO add bool for inactive vehicles that are not in operation anymore (either time ran out or returned to depot)
        # recreated by VehicleHandler.add_manifest_to_vehicles()
        self.started = False
        self.rebalancing = False
        self.dwelling = False
        self.time_at_last = start_time # this is definitely incorrect if a vehicle is recreated in the offlineSolver and a vehicle has already been boarded
        self.last_node = start_node
        self.time_at_next = start_time
        self.next_immediate_node = start_node
        self.trips: dict[int, Trip] = {}
        self.picked = []
        self.served_trips = []
        self.stop_sequence: list[VehicleStop] = [] # TODO add StopSequence wherever this list is updated or used
        self.final_stop_time = start_time # TODO this is never updated

        # for ML features, add entire manifest as we have more flexibility in creating features in relation to other state or trip features
        self._manifest: list[ManifestEntry] = manifest 
        # vehicle-state capacities are the current capacities based on taken trips and not the original full ones; we do not change anything in the simulation as the side effects are unknown, added new full vehicle capacities to have that information ready
        self.full_am_capacity = am_capacity
        self.full_wc_capacity = wc_capacity
        # TODO validate whether the entries are all correct use this to fill the other state variables, such as picked and sequence because we currently do not consider them, but they should be easily overridden as we want to update the vehicle_state freely

    @property
    def manifest(self):
        # we do not wnat to change the manifest for now, but rather have the information on hand
        return self._manifest

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
    
    def __str__(self):
        trip_str = "{" + ', '.join([f"{trip_id}: {str(trip)}" for trip_id, trip in self.trips.items()]) + "}"
        picked_str = ', '.join([str(pick) for pick in self.picked])
        served_str = ', '.join([str(served) for served in self.served_trips])
        return f"Vehicle ID {self.id}: time: {self.start_time}>{self.end_time}, trips: {trip_str}, picked: [{picked_str}], served: [{served_str}], last_node: {self.last_node}, stop_sequence: [{self.stop_sequence}]"
    
    def __repr__(self):
        return (
            f"Vehicle("
            f"vehicle_id={self.id!r}, "
            f"start_node={self.next_immediate_node!r}, " # possibly last_node makes more sense
            f"am_capacity={self.am_capacity!r}, "
            f"wc_capacity={self.wc_capacity!r}, "
            f"start_time={self.start_time!r}, "
            f"end_time={self.end_time!r}, "
            f"depot={self.depot!r}"
            f")"
        )
    
    def to_dict(self) -> dict:
        """
        get the entire vehicle including the current state variables instead of adapting their creation.
        """
        return {
            # Static vehicle information
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "depot": self.depot,
            "am_capacity": self.am_capacity,
            "wc_capacity": self.wc_capacity,

            # Simulation state
            "started": self.started,
            "rebalancing": self.rebalancing,
            "dwelling": self.dwelling,
            "time_at_last": self.time_at_last,
            "last_node": self.last_node,
            "time_at_next": self.time_at_next,
            "next_immediate_node": self.next_immediate_node,

            # Trips
            "trips": {
                trip_id: trip.to_dict() if hasattr(trip, "to_dict") else trip
                for trip_id, trip in self.trips.items()
            },

            "picked": list(self.picked),
            "served_trips": list(self.served_trips),

            # Stops
            "stop_sequence": [
                stop.to_dict() if hasattr(stop, "to_dict") else stop
                for stop in self.stop_sequence
            ],

            "final_stop_time": self.final_stop_time,
            "manifest": self._manifest
        }

    @classmethod
    def from_dict(cls, data: dict):
        # TODO not yet tested and uncertain whether it is required, if we create a new object from this, the changes would not apply anymore to the actual vehicle
        obj = cls(
            vehicle_id  =       data["id"],
            start_time =        data["start_time"],
            end_time =          data["end_time"],
            depot =             data["depot"],
            am_capacity =       data["am_capacity"],
            wc_capacity =       data["wc_capacity"],
            start_node =        data["last_node"],  # or depot depending on design
            manifest =          data["manifest"]
        )

        # Restore simulation state
        obj.started =           data["started"]
        obj.rebalancing =       data["rebalancing"]
        obj.dwelling =          data["dwelling"]
        obj.time_at_last =      data["time_at_last"]
        obj.last_node =         data["last_node"]
        obj.time_at_next =      data["time_at_next"]
        obj.next_immediate_node = data["next_immediate_node"]
        obj.trips =             data["trips"]
        obj.picked =            data["picked"]
        obj.served_trips =      data["served_trips"]
        obj.stop_sequence =     data["stop_sequence"]
        obj.final_stop_time =   data["final_stop_time"]

        return obj

    # HELPER METHODS TO CREATE ML FEATURES
    def get_capacities(self) -> tuple[int, int, int, int]:
        """
        Get used and full capacities for both am and wc 
        
        :return: used capacities for 'am' and 'wc', real capacities for 'am' and 'wc'
        :rtype: tuple[int, int, int, int]
        """
        if self.picked:
            am_used = 0
            wc_used = 0
            am_capacity = self.am_capacity
            wc_capacity = self.wc_capacity
            matching_entries = []
            for trip_id in self.picked:
                iteration, booking_id = self._tripId_split_to_bookingID(trip_id)
                matching_entries.extend([entry for entry in self.manifest if entry.booking_id == booking_id])
                assert matching_entries, "Why are matching_entries empty?"

            for entry in matching_entries:
                if entry.action == VehicleStop.ACT_DROPOFF:
                    am_used += entry.am
                    wc_used += entry.wc
                    am_capacity += entry.am
                    wc_capacity += entry.wc
                    assert am_used < 8 and am_used >= 0, f"Used am-capacity {am_used} at cap {self.am_capacity}do not make sense." # 8 seems default
                    assert wc_used < 3 and wc_used >= 0, f"Used wc-capacity {wc_used} at cap {self.wc_capacity}do not make sense." # 3 seems default

            return am_used, wc_used, am_capacity, wc_capacity
        return 0, 0, self.am_capacity, self.wc_capacity
    
    def get_remaining_capacities(self):
        """return remaining caps normalized"""
        am_used, wc_used, am_cap, wc_cap = self.get_capacities()
        remaining_am_cap = (am_cap - am_used) / am_cap
        remaining_wc_cap = (wc_cap - wc_used) / wc_cap
        return am_cap, wc_cap, remaining_am_cap, remaining_wc_cap
    
    def get_remaining_boarded_time(self, current_time: float) -> float:
        """
        Get remaining time until currently boarded requests are last dropped
        
        :return: Time to drop last picked request
        :rtype: float
        """
        if self.picked:
            for trip_id in self.picked:
                # trip_id of structure '2-1' for iteration-booking_id
                iteration, booking_id = self._tripId_split_to_bookingID(trip_id)
                matching_entries = [entry for entry in self._manifest if entry.booking_id == booking_id]
                assert matching_entries, "Why are matching_entries empty?"

            scheduled_times = []
            for entry in matching_entries:
                scheduled_times.append(entry.scheduled_time) # as each pickup has a dropoff, the later time must be a dropoff
            last_boarded_dropoff_time = max(scheduled_times)
            assert last_boarded_dropoff_time > current_time, f"Last dropoff is not bigger than current_time: {last_boarded_dropoff_time} >X {current_time}"
            return last_boarded_dropoff_time
        return 0
    
    @staticmethod
    def _tripId_split_to_bookingID(trip_id: str) -> tuple[int, int]:
        split_id = trip_id.split("-")
        iteration, booking_id = int(split_id[0]), int(split_id[1])
        return iteration, booking_id
 
# BELOW DEPRECATED METHODS FOR SIMULATION
    def has_completed_operations(self, current_time: int):
        """deprecated; boolean for checking whether the vehicle's operational time has run out and the vehicle is empty."""
        if len(self.stop_sequence) == 0 and self.end_time <= current_time:
            assert len(self.picked) == 0
            return True
        return False

    def return_to_depot(self) -> VehicleStop:
        """deprecated; handle steps to build stop to return to depot."""
        # TODO FIXME this currently does not work, the final_stop_time does not seem to be correct as it is never updated
        stop = VehicleStop(None, self.depot, VehicleStop.ACT_DEPOT, 0)
        stop.stop_time = self.final_stop_time + NetworkHandler.travel_time(self.last_node, self.depot)
        stop.vehicle_id = self.id 

        return stop
    
    def restore_state_from_manifest(self, driver_run, boarded_requests, boarded_trips, dwell_alight, dwell_pickup):
        """
        @deprecated

        TODO interim solution: add manifest to vehicle in order to retrieve actual features from the vehicle; we need something like this as we currently always initialize a new object only from the basic information instead of the full set, including current_location, next_location etc.
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

   