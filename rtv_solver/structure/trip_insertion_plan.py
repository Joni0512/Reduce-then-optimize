from dataclasses import dataclass

from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.trip import Trip
from rtv_solver.structure.node import Node


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
