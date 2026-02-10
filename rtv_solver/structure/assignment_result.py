from dataclasses import dataclass

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.node import Node

@dataclass
class AssignmentResult:
    vehicle_assignment: dict[int, tuple[list[Trip], list[int]]]     # {vehicle_id: ([trips], StopSequence)}   
    request_assignment: dict[int, int]                              # {request_id: vehicle_id}
    rebalancing_assignment: dict[int, Node]                         # {vehicle_id: origin_node}

    unassigned_trip_count: int
    taxi_only_trip_count: int

    added_distance: float
    trip_sizes: list[int]

    status: int
    runtime: float | None = None