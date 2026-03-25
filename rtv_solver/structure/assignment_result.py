from __future__ import annotations

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

    @classmethod
    def get_empty_result(cls, *, unassigned_trip_count: int) -> AssignmentResult:
        """No vehicles or nothing to assign: valid result with empty assignments.
        Status 5 is created to be differentiated."""
        return cls(
            vehicle_assignment={},
            request_assignment={},
            rebalancing_assignment={},
            unassigned_trip_count=unassigned_trip_count,
            taxi_only_trip_count=0,
            added_distance=0.0,
            trip_sizes=[],
            status=5,
        )

    def __repr__(self) -> str:
        return (
            "AssignmentResult("
            f"status={self.status!r}, "
            f"runtime={self.runtime!r}, "
            f"vehicle_assignment={self.vehicle_assignment!r}, "
            f"request_assignment={self.request_assignment!r}, "
            f"rebalancing_assignment={self.rebalancing_assignment!r}, "
            f"unassigned_trip_count={self.unassigned_trip_count}, "
            f"taxi_only_trip_count={self.taxi_only_trip_count}, "
            f"added_distance={self.added_distance}, "
            f"trip_sizes={self.trip_sizes}"
            ")"
        )