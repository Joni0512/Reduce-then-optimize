import pytest
import numpy as np
import torch

from rtv_solver.pipeline import CO_TripCostMinimization, CO_ScoreMaximization
from rtv_solver.structure.assignment_result import AssignmentResult

from rtv_solver.structure.vehicle import Vehicle

@pytest.mark.basic
def test_co_run_trip_cost_minimization(
    single_trip_map, 
    trip_list, 
    trip_costs, 
    vehicle_to_trips_cost_map, 
    trip_to_vehicle_cost_map, 
    config, 
    request_batch, 
    active_requests):
    """check whether CO layer returns correct results from the outputs of a tripHandler"""

    optimizer = CO_TripCostMinimization(config)
    optimizer.reset(single_trip_map, trip_list, trip_costs, vehicle_to_trips_cost_map, trip_to_vehicle_cost_map)
    result = optimizer.run(request_batch, active_requests)

    assert isinstance(result, AssignmentResult)
    assert result.unassigned_trip_count == 3
    assert result.taxi_only_trip_count == 2
    assert result.added_distance == pytest.approx(798.3)
    assert result.trip_sizes == [2]
    assert result.status == 2
    assert result.rebalancing_assignment == {}
    
    assert result.request_assignment == {1: 0, 5: 0}
    assert len(result.request_assignment) == 2
    
    assert len(result.vehicle_assignment) == 1
    assert 0 in result.vehicle_assignment

    trips, stops = result.vehicle_assignment[0]

    assert len(trips) == 2
    assert len(stops) == 4

    t1 = trips[0]

    assert t1.request_id == 1
    assert t1.number == 0
    assert t1.am_capacity == 1
    assert t1.wc_capacity == 0
    assert t1.pick_up_time == 19822
    assert t1.latest_pick_up_time == 21622
    assert t1.earliest_arrival_time == 20112.7
    assert t1.latest_arrival_time == 21912.7
    assert t1.dwell_pickup == 180
    assert t1.dwell_alight == 60
    assert t1.iteration == 2

    assert t1.origin.lat == 35.707904816
    assert t1.origin.lon == -77.90247345
    assert t1.destination.lat == 35.737380981
    assert t1.destination.lon == -77.906433105

    t2 = trips[1]

    assert t2.request_id == 5
    assert t2.number == 4
    assert t2.am_capacity == 1
    assert t2.wc_capacity == 0
    assert t2.pick_up_time == 20002
    assert t2.latest_pick_up_time == 21802
    assert t2.earliest_arrival_time == 20451.0
    assert t2.latest_arrival_time == 22251.0
    assert t2.dwell_pickup == 180
    assert t2.dwell_alight == 60
    assert t2.iteration == 2

    assert t2.origin.lat == 35.720035553
    assert t2.origin.lon == -77.893722534
    assert t2.destination.lat == 35.753082275
    assert t2.destination.lon == -77.930335999

    s1, s2, s3, s4 = stops

    assert s1.trip_id == "2-1"
    assert s1.type == "pickup"
    assert s1.dwell == 180

    assert s2.trip_id == "2-5"
    assert s2.type == "pickup"
    assert s2.dwell == 180

    assert s3.trip_id == "2-1"
    assert s3.type == "dropoff"
    assert s3.dwell == 60

    assert s4.trip_id == "2-5"
    assert s4.type == "dropoff"
    assert s4.dwell == 60

    assert result.runtime >= 0


@pytest.mark.basic
def test_co_run_trip_cost_minimization_neworder(single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder, request_batch, active_requests, config):
    """check whether CO layer returns correct results from the outputs of a tripHandler; trip_costs are specifically changed to not have perfect ordering"""

    optimizer = CO_TripCostMinimization(config)
    optimizer.reset(single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder)
    result = optimizer.run(request_batch, active_requests)

    assert isinstance(result, AssignmentResult)
    assert result.unassigned_trip_count == 3
    assert result.taxi_only_trip_count == 2
    assert result.added_distance == pytest.approx(798.3)
    assert result.trip_sizes == [2]
    assert result.status == 2
    assert result.rebalancing_assignment == {}
    
    assert result.request_assignment == {1: 0, 5: 0}
    assert len(result.request_assignment) == 2
    
    assert len(result.vehicle_assignment) == 1
    assert 0 in result.vehicle_assignment

    trips, stops = result.vehicle_assignment[0]

    assert len(trips) == 2
    assert len(stops) == 4

    t1 = trips[0]

    assert t1.request_id == 1
    assert t1.number == 0
    assert t1.am_capacity == 1
    assert t1.wc_capacity == 0
    assert t1.pick_up_time == 19822
    assert t1.latest_pick_up_time == 21622
    assert t1.earliest_arrival_time == 20112.7
    assert t1.latest_arrival_time == 21912.7
    assert t1.dwell_pickup == 180
    assert t1.dwell_alight == 60
    assert t1.iteration == 2

    assert t1.origin.lat == 35.707904816
    assert t1.origin.lon == -77.90247345
    assert t1.destination.lat == 35.737380981
    assert t1.destination.lon == -77.906433105

    t2 = trips[1]

    assert t2.request_id == 5
    assert t2.number == 4
    assert t2.am_capacity == 1
    assert t2.wc_capacity == 0
    assert t2.pick_up_time == 20002
    assert t2.latest_pick_up_time == 21802
    assert t2.earliest_arrival_time == 20451.0
    assert t2.latest_arrival_time == 22251.0
    assert t2.dwell_pickup == 180
    assert t2.dwell_alight == 60
    assert t2.iteration == 2

    assert t2.origin.lat == 35.720035553
    assert t2.origin.lon == -77.893722534
    assert t2.destination.lat == 35.753082275
    assert t2.destination.lon == -77.930335999

    s1, s2, s3, s4 = stops

    assert s1.trip_id == "2-1"
    assert s1.type == "pickup"
    assert s1.dwell == 180

    assert s2.trip_id == "2-5"
    assert s2.type == "pickup"
    assert s2.dwell == 180

    assert s3.trip_id == "2-1"
    assert s3.type == "dropoff"
    assert s3.dwell == 60

    assert s4.trip_id == "2-5"
    assert s4.type == "dropoff"
    assert s4.dwell == 60

    assert result.runtime >= 0


@pytest.mark.basic
def test_co_run_score_maximization(single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder, request_batch, active_requests, config):
    """check whether CO layer with new feature scores returns correct matching based on the feature scores"""

    feature_scores = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

    optimizer = CO_ScoreMaximization(config)
    optimizer.reset(single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder)
    result = optimizer.run(feature_scores, request_batch, active_requests)

    assert isinstance(result, AssignmentResult)
    assert result.unassigned_trip_count == 4
    assert result.taxi_only_trip_count == 1
    assert result.status == 2
    assert result.rebalancing_assignment == {}
    assert result.request_assignment == {1: 0}
    assert len(result.request_assignment) == 1
    
    assert len(result.vehicle_assignment) == 1
    assert 0 in result.vehicle_assignment

    trips, stops = result.vehicle_assignment[0]

    assert len(trips) == 1
    assert len(stops) == 2

    assert trips == [0]


@pytest.mark.basic
def test_transform_optimal_solution_to_assignment_handles_reject_action(
    single_trip_map_neworder,
    trip_list_neworder,
    trip_costs_neworder,
    vehicle_to_trips_cost_map_neworder,
    trip_to_vehicle_cost_map_neworder,
    request_batch,
    config,
):
    """
    Decode-only test: no model scoring and no ILP solve.
    Ensure a selected reject action for a vehicle yields no trip assignment.
    """
    optimizer = CO_ScoreMaximization(config)
    optimizer.reset(
        single_trip_map_neworder,
        trip_list_neworder,
        trip_costs_neworder,
        vehicle_to_trips_cost_map_neworder,
        trip_to_vehicle_cost_map_neworder,
    )

    # 9 trip entries + 1 reject entry for vehicle 0.
    y_star = torch.zeros(len(trip_costs_neworder) + 1, dtype=torch.float32)
    y_star[-1] = 1.0

    result = optimizer.transform_optimal_solution_to_assignment(
        y_star=y_star,
        requests=request_batch,
        reject_vehicle_ids=[0],
    )

    assert isinstance(result, AssignmentResult)
    assert result.vehicle_assignment == {}
    assert result.request_assignment == {}
    assert result.unassigned_trip_count == len(request_batch)
    assert result.taxi_only_trip_count == 0
    assert result.added_distance == pytest.approx(0.0)
    assert result.trip_sizes == []
    assert result.status == 2


@pytest.mark.basic
def test_co_run_score_maximization_prefers_reject_action(
    single_trip_map_neworder,
    trip_list_neworder,
    trip_costs_neworder,
    vehicle_to_trips_cost_map_neworder,
    trip_to_vehicle_cost_map_neworder,
    request_batch,
    active_requests,
    config,
):
    """
    Optimization-level reject test with fake scores:
    all trip scores are negative while reject score is neutral, so reject should
    be selected for the vehicle.
    """
    optimizer = CO_ScoreMaximization(config)
    optimizer.reset(
        single_trip_map_neworder,
        trip_list_neworder,
        trip_costs_neworder,
        vehicle_to_trips_cost_map_neworder,
        trip_to_vehicle_cost_map_neworder,
    )

    feature_scores = np.full(len(trip_costs_neworder), -1.0, dtype=float)
    reject_action_scores = np.array([0.0], dtype=float)
    reject_vehicle_ids = [0]

    result = optimizer.run(
        feature_scores,
        request_batch,
        active_requests,
        reject_action_scores=reject_action_scores,
        reject_vehicle_ids=reject_vehicle_ids,
    )

    assert isinstance(result, AssignmentResult)
    assert result.status == 2
    assert result.vehicle_assignment == {}
    assert result.request_assignment == {}
    assert result.unassigned_trip_count == len(request_batch)
    assert result.taxi_only_trip_count == 0
    assert result.added_distance == pytest.approx(0.0)
    assert result.trip_sizes == []


@pytest.mark.basic
@pytest.mark.parametrize(
    "feature_scores,expected_trip_indices",
    [   
        pytest.param(
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), [0],
            id = "co_score-1"
        ),
        pytest.param(
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]), [1],
            id = "co_score-2"
        ),
        pytest.param(
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]), [0, 3],
            id = "co_score-combined"
        ),
    ]
)
def test_co_run_score_maximization(feature_scores, expected_trip_indices,single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder, request_batch, active_requests, config):
    """check whether CO layer with new feature scores returns correct matching based on the feature scores"""
    optimizer = CO_ScoreMaximization(config)
    optimizer.reset(single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder)
   
    result = optimizer.run(feature_scores, request_batch, active_requests)

    assigned_count = len(expected_trip_indices)
    assert isinstance(result, AssignmentResult)
    assert result.unassigned_trip_count == 5 - assigned_count
    assert result.taxi_only_trip_count == assigned_count
    assert result.status == 2
    assert len(result.request_assignment) == assigned_count
    
    assert len(result.vehicle_assignment) == 1
    assert 0 in result.vehicle_assignment

    trips, stops = result.vehicle_assignment[0]

    assert len(trips) == assigned_count
    assert len(stops) == assigned_count * 2

    for i, trip in enumerate(trips):
        assert trip.number == expected_trip_indices[i]

# TODO test assignment with multiple vehicles for scores (requires some efforts for the test setup as we need to export the required information from the tripHandler to simulate the CO layer, interface far too complex)

    
@pytest.mark.basic
def test_apply_trip_insertion(vehicle_init: Vehicle, trip_insertion_plan):
    vehicle_init.trips = {}
    vehicle_init.stop_sequence = []
    vehicle_init.time_at_last = 0
    vehicle_init.time_at_next = 0
    vehicle_init.last_node = None
    vehicle_init.rebalancing = True
    
    plan = trip_insertion_plan
    vehicle_init.apply_trip_insertion(plan)

    assert vehicle_init.rebalancing is False
    assert vehicle_init.last_node == plan.next_immediate_node
    assert vehicle_init.time_at_last == plan.time_at_next_immediate_node
    assert vehicle_init.time_at_next >= plan.time_at_next_immediate_node + plan.veh_travel_time

    assert set(vehicle_init.trips.keys()) == {trip.id for trip in plan.trips}
    assert vehicle_init.stop_sequence == plan.sequence

    first_stop = vehicle_init.stop_sequence[0]
    first_trip = vehicle_init.trips[first_stop.trip_id]
    if first_stop.type == first_stop.ACT_PICKUP:
        assert vehicle_init.time_at_next >= first_trip.pick_up_time
    elif first_stop.type == first_stop.ACT_DROPOFF:
        assert vehicle_init.time_at_next >= first_trip.earliest_arrival_time