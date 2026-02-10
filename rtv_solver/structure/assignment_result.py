from dataclasses import dataclass

from rtv_solver.structure.trip import Trip

@dataclass
class AssignmentResult:
    vehicle_assignment: dict[int, tuple[list[Trip], list[int]]] 
    request_assignment: dict[int, int]

    unassigned_trip_count: int
    taxi_only_trip_count: int

    added_distance: float
    trip_sizes: list[int]

    status: int
    runtime: float | None = None