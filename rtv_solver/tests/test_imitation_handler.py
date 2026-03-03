import pytest

from rtv_solver.pipeline.imitation_handler import ImitationHandler
import numpy as np

@pytest.mark.basic
def test_score_complete_ordered_match_returns_1000():
    optimal_solution = [1, 3, 2, 4, 7]
    combinations = [(1, 3, 2, 4, 7)]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    assert scores == [1000]


@pytest.mark.basic
def test_score_ordered_prefix_partial_match_gets_positive_length_bonus():
    optimal_solution = [1, 3, 2, 4, 7]
    combinations = [(1, 3, 2)]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    # starts at index 0 -> 100 + 10 * len(combination)
    assert scores == [130]


@pytest.mark.basic
def test_score_ordered_non_prefix_partial_match_gets_negative_length_penalty():
    optimal_solution = [1, 3, 2, 4, 7]
    combinations = [(2, 4, 7)]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    # starts at index > 0 -> -10 * len(combination)
    assert scores == [-30]


@pytest.mark.basic
def test_score_unordered_full_window_match_scales_with_length():
    optimal_solution = [1, 3, 2, 4, 7]
    combinations = [(3, 1)]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    # unordered full window match -> -100 - 10 * len(combination)
    assert scores == [-120]


@pytest.mark.basic
def test_score_no_match_returns_zero():
    optimal_solution = [1, 3, 2, 4, 7]
    combinations = [(8, 9)]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    assert scores == [0]


@pytest.mark.basic
def test_score_multiple_combinations_returns_expected_vector():
    optimal_solution = [1, 3, 2, 4, 7]
    combinations = [
        (1, 3, 2, 4, 7),  # complete ordered match
        (1, 3, 2),        # ordered prefix
        (2, 4),           # ordered non-prefix
        (3, 1),           # unordered full-window match
        (8, 9),           # no match
    ]

    scores = ImitationHandler.score_combinations_against_solution(
        combinations, optimal_solution
    )

    assert scores == [1000, 130, -20, -120, 0]


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
    assert (0, 1) in combinations
    assert (1, 2, 3) in combinations


@pytest.mark.basic
@pytest.mark.parametrize(
    "solution,max_value,min_value",
    [
        pytest.param([1, 3, 2, 4, 7],120,-150),
        pytest.param([1, 2, 3, 4, 5],1000,-150)
    ])
def test_get_trip_index_combinations_returns_expected_combinations(solution, max_value, min_value):
    trips = [0, 1, 2, 3, 4, 5, 6, 7]

    combinations, counts = ImitationHandler.get_trip_index_combinations(
        trips=trips)
    scores = ImitationHandler.score_combinations_against_solution(
        combinations, solution)

    np_scores = np.array(scores)
    max_idx = np.argmax(np_scores)
    min_idx = np.argmin(np_scores)
    assert np_scores[max_idx] == max_value
    assert np_scores[min_idx] == min_value