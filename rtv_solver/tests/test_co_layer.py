import pytest

from rtv_solver.pipeline.co_tripCostMinimization import CO_TripCostMinimization
from rtv_solver.structure.assignment_result import AssignmentResult

@pytest.mark.basic
def test_combinatorial_optimization_run(
    single_trip_map, 
    trip_list, 
    trip_costs, 
    vehicle_to_trips_cost_map, 
    trip_to_vehicle_cost_map, 
    config, 
    request_batch, 
    active_requests):
    """check whether CO layer returns correct results from the outputs of a tripHandler"""

    optimizer = CO_TripCostMinimization(
        single_trip_map, 
        trip_list, 
        trip_costs,
        vehicle_to_trips_cost_map, 
        trip_to_vehicle_cost_map, 
        config)
    result = optimizer.run(request_batch, active_requests)

    # TODO implement equalities to compare these results
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
def test_combinatorial_optimization_run_neworder(single_trip_map_neworder, trip_list_neworder, trip_costs_neworder, vehicle_to_trips_cost_map_neworder, trip_to_vehicle_cost_map_neworder, request_batch, active_requests, config):
    """check whether CO layer returns correct results from the outputs of a tripHandler; trip_costs are specifically changed to not have perfect ordering"""

    optimizer = CO_TripCostMinimization(
        single_trip_map_neworder, 
        trip_list_neworder, 
        trip_costs_neworder,
        vehicle_to_trips_cost_map_neworder, 
        trip_to_vehicle_cost_map_neworder, 
        config)
    result = optimizer.run(request_batch, active_requests)

    # TODO implement equalities to compare these results
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
def test_apply_trip_insertion(vehicle_init, trip_insertion_plan):
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