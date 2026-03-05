import pytest
import torch

from rtv_solver.pipeline import ImitationHandler

@pytest.mark.basic
@pytest.mark.parametrize(
    "optimal_solution,combinations,expected_scores",
    [
        pytest.param([1, 3, 2], [(1, 3, 2)], torch.tensor([1000])),
        pytest.param([1, 3, 2], [(1, 2, 3)], torch.tensor([0])),
        pytest.param([1, 3, 2], [(2, 4, 7)], torch.tensor([0])),
        pytest.param([1, 3, 2], [(1, 3)], torch.tensor([120])), 
        pytest.param([2, 1], [(2,)], torch.tensor([110])),
    ]
) 
def test_score_order_variations(optimal_solution, combinations, expected_scores):
    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )
    assert scores == expected_scores

@pytest.mark.basic
def test_score_multiple_combinations_returns_expected_vector():
    optimal_solution = [10, 30, 20, 40, 70]
    combinations = [
        (10, 30, 20, 40, 70),  # complete ordered match
        (10, 30, 20),        # ordered prefix
        (20, 40),           # ordered non-prefix
        (30, 10),           # unordered full-window match
        (80, 90),           # no match
    ]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    assert torch.equal(scores, torch.tensor([1000, 130, 0, 0, 0]))


@pytest.mark.basic
def test_get_trip_index_combinations_returns_counts_by_cardinality():
    trips = [10, 20, 30, 40]

    combinations, counts = ImitationHandler.get_trip_index_combinations(
        trips=trips,
        min_cardinality=2,
        max_cardinality=3,
    )

    # C(4,2)=6 and C(4,3)=4
    assert counts == {2: 6, 3: 4}
    assert len(combinations) == 10
    assert (10, 20) in combinations
    assert (20, 30, 40) in combinations

@pytest.mark.basic
@pytest.mark.parametrize(
    "solution,max_value,min_value",
    [
        pytest.param([10, 30, 20, 40, 70],torch.tensor(120),torch.tensor(0)),
        pytest.param([10, 20, 30, 40, 50],torch.tensor(1000),torch.tensor(0)),
        pytest.param([0, 30, 20, 40, 70],torch.tensor(120),torch.tensor(0)),
        pytest.param([0, 30, 40, 80],torch.tensor(130),torch.tensor(0)),
        pytest.param([80, 30, 20, 40, 70],torch.tensor(0),torch.tensor(0)),
        pytest.param([1, 2, 3],torch.tensor(0),torch.tensor(0)),
    ])
def test_score_max_min_values(solution, max_value, min_value):
    trips = [0, 10, 20, 30, 40, 50, 60, 70]

    combinations, counts = ImitationHandler.get_trip_index_combinations(
        trips=trips)
    scores = ImitationHandler.score_combinations_against_solution(
        combinations, solution)

    max_idx = torch.argmax(scores)
    min_idx = torch.argmin(scores)
    assert scores[max_idx] == max_value
    assert scores[min_idx] == min_value


@pytest.mark.basic
def test_append_reject_action_scores_per_vehicle_penalty_logic():
    # Trip scores by vehicle:
    # v0 -> [120, 0] has a positive score => reject score 0
    # v1 -> [0, 0]   no positive score     => reject score -10
    imitation_scores = torch.tensor([120.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    trip_vehicle_ids = [0, 0, 1, 1]
    reject_vehicle_ids = [0, 1]

    out = ImitationHandler.append_reject_action_scores(
        imitation_scores,
        reject_vehicle_ids,
        trip_vehicle_ids=trip_vehicle_ids,
    )
    assert torch.equal(out, torch.tensor([120.0, 0.0, 0.0, 0.0, 0.0, -10.0]))


@pytest.mark.basic
def test_append_reject_action_scores_global_fallback_without_mapping():
    imitation_scores = torch.tensor([0.0, 0.0], dtype=torch.float32)
    reject_vehicle_ids = [0, 1]

    out = ImitationHandler.append_reject_action_scores(
        imitation_scores,
        reject_vehicle_ids,
        trip_vehicle_ids=None,
    )
    assert torch.equal(out, torch.tensor([0.0, 0.0, -10.0, -10.0]))


@pytest.mark.basic
def test_append_reject_action_scores_raises_on_mapping_length_mismatch():
    imitation_scores = torch.tensor([1.0, 0.0], dtype=torch.float32)
    reject_vehicle_ids = [0]
    trip_vehicle_ids = [0]

    with pytest.raises(ValueError):
        ImitationHandler.append_reject_action_scores(
            imitation_scores,
            reject_vehicle_ids,
            trip_vehicle_ids=trip_vehicle_ids,
        )