from types import SimpleNamespace

import pytest
import torch

from rtv_solver.pipeline import ImitationHandler


class FakeTripCost:
    """
    Minimal TripCost test double for isolated imitation-handler tests.

    Only the two attributes used by the tested methods are implemented:
    - `vehicle_id`
    - `get_ordered_request_ids()`
    """

    def __init__(self, vehicle_id: int, request_ids: tuple[int, ...]):
        self.vehicle_id = vehicle_id
        self._request_ids = request_ids

    def get_ordered_request_ids(self) -> tuple[int, ...]:
        return self._request_ids


def _handler_with_optimal_solution(optimal_solution: dict[int, list[int]]) -> ImitationHandler:
    """
    Construct an ImitationHandler instance without reading payload files.

    Tests can inject a precise per-vehicle optimal sequence map directly.
    """
    handler = ImitationHandler.__new__(ImitationHandler)
    handler.optimal_solution = optimal_solution
    return handler


@pytest.mark.basic
def test_get_optimal_request_solution_for_batch_by_vehicle_preserves_order():
    handler = _handler_with_optimal_solution(
        {
            0: [10, 30, 50],
            1: [20, 40, 60],
        }
    )
    request_batch = [
        SimpleNamespace(id=60),
        SimpleNamespace(id=10),
        SimpleNamespace(id=40),
        SimpleNamespace(id=30),
        SimpleNamespace(id=70),
    ]

    out = handler.get_optimal_request_solution_for_batch_by_vehicle(
        request_batch=request_batch,
        vehicle_ids=[0, 1],
    )
    assert out == {0: [10, 30], 1: [40, 60]}


@pytest.mark.basic
def test_build_trip_vehicle_index_map_interleaved_trip_order():
    trip_costs = [
        FakeTripCost(vehicle_id=1, request_ids=(20,)),
        FakeTripCost(vehicle_id=0, request_ids=(10, 30)),
        FakeTripCost(vehicle_id=1, request_ids=(20, 40)),
        FakeTripCost(vehicle_id=0, request_ids=(10,)),
    ]

    out = ImitationHandler.build_trip_vehicle_index_map(trip_costs)
    assert out == {1: [0, 2], 0: [1, 3]}


@pytest.mark.basic
def test_score_trip_costs_against_optimal_by_vehicle_keeps_global_alignment():
    """
    Scores are computed per vehicle and written back to original global indices.
    """
    handler = _handler_with_optimal_solution(
        {
            0: [10, 30],
            1: [20, 40],
        }
    )
    trip_costs = [
        FakeTripCost(vehicle_id=1, request_ids=(20,)),
        FakeTripCost(vehicle_id=0, request_ids=(10, 30)),
        FakeTripCost(vehicle_id=1, request_ids=(20, 40)),
        FakeTripCost(vehicle_id=0, request_ids=(10,)),
    ]
    request_batch = [
        SimpleNamespace(id=10),
        SimpleNamespace(id=20),
        SimpleNamespace(id=30),
        SimpleNamespace(id=40),
    ]
    # Explicit map simulates the structure provided by TripHandler.
    vehicle_to_trips_cost_map = {0: [1, 3], 1: [0, 2]}

    scores, trip_vehicle_ids = handler.score_trip_costs_against_optimal_by_vehicle(
        trip_costs=trip_costs,
        request_batch=request_batch,
        vehicle_to_trips_cost_map=vehicle_to_trips_cost_map,
    )

    # Global index expectations:
    # idx0 (v1, (20,))     -> prefix score 110
    # idx1 (v0, (10, 30))  -> full exact score 1000
    # idx2 (v1, (20, 40))  -> full exact score 1000
    # idx3 (v0, (10,))     -> prefix score 110
    assert torch.equal(scores, torch.tensor([110, 1000, 1000, 110]))
    assert trip_vehicle_ids == [1, 0, 1, 0]


@pytest.mark.basic
def test_build_y_star_per_vehicle_from_scores_raises_on_non_zero_max_tie():
    """
    If a vehicle has multiple equal non-zero maxima, this is considered ambiguous supervision and must raise.
    """
    scores = torch.tensor([5.0, 4.0, 5.0, 1.0, -10.0, -10.0])
    trip_vehicle_ids = [1, 0, 1, 0]
    reject_vehicle_ids = [0, 1]

    with pytest.raises(ValueError, match="multiple non-zero maxima"):
        ImitationHandler.build_y_star_per_vehicle_from_imit_scores(
            imitation_scores_with_reject=scores,
            trip_vehicle_ids=trip_vehicle_ids,
            reject_vehicle_ids=reject_vehicle_ids,
        )


@pytest.mark.basic
def test_build_y_star_per_vehicle_from_scores_all_non_positive_prefers_minimum():
    """
    For all non-positive candidates, each vehicle chooses its minimum score.
    With reject=-10 this selects reject slots for both vehicles.
    """
    scores = torch.tensor([-1.0, -2.0, -3.0, -4.0, -10.0, -10.0])
    trip_vehicle_ids = [1, 0, 1, 0]
    reject_vehicle_ids = [0, 1]

    y_star = ImitationHandler.build_y_star_per_vehicle_from_imit_scores(
        imitation_scores_with_reject=scores,
        trip_vehicle_ids=trip_vehicle_ids,
        reject_vehicle_ids=reject_vehicle_ids,
    )

    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.equal(y_star, expected)


@pytest.mark.basic
def test_build_y_star_per_vehicle_from_scores_equal_max_across_vehicles_is_valid():
    """
    Equal maxima across different vehicles are valid because each vehicle has
    its own independent candidate set.
    """
    scores = torch.tensor([5.0, 5.0, 1.0, 2.0, -10.0, -10.0])
    trip_vehicle_ids = [1, 0, 1, 0]
    reject_vehicle_ids = [0, 1]

    y_star = ImitationHandler.build_y_star_per_vehicle_from_imit_scores(
        imitation_scores_with_reject=scores,
        trip_vehicle_ids=trip_vehicle_ids,
        reject_vehicle_ids=reject_vehicle_ids,
    )

    # Vehicle 1 selects idx0 (score 5), vehicle 0 selects idx1 (score 5).
    expected = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.equal(y_star, expected)
