import pytest

from rtv_solver.pipeline.optimizer import CO_TripCostMinimization
from rtv_solver.structure.assignment_result import AssignmentResult
from rtv_solver.handlers.vehicle_handler import VehicleHandler

class FakeVehicleHandler:
    def __init__(self, vehicles):
        self.vehicles = vehicles

@pytest.mark.basic
def test_combinatorial_optimization_run(single_trip_map, 
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

    # AssignmentResult(vehicle_assignment={0: ([Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, ), Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, )], [VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)])}, request_assignment={1: 0, 5: 0}, rebalancing_assignment={}, unassigned_trip_count=3, taxi_only_trip_count=2, added_distance=798.3, trip_sizes=[2], status=2, runtime=0.0037648677825927734)

@pytest.mark.basic
def test_plan_creation(request_batch, assignment_result, vehicle):
    # FIXME fix the following test
    result = assignment_result
    vehicle_handler = FakeVehicleHandler({0: vehicle})

    unserved_requests = set([req.id for req in request_batch]) # number of requests that are not already confirmed to be  served
    for vehicle_id in result.vehicle_assignment: # if it is empty the assignment is skipped
        vehicle = vehicle_handler.vehicles[vehicle_id]
        trips, prev_sequence = result.vehicle_assignment[vehicle_id]
        plan = VehicleHandler.plan_trip_insertions(vehicle, trips, prev_sequence=prev_sequence)
        # vehicle.apply_trip_insertion(plan) NOT TESTED here
        for trip in trips: # remove assigned trips from unserved
            if trip.request_id in unserved_requests:
                unserved_requests.remove(trip.request_id)
    assert unserved_requests == {2, 3, 4}

    assert plan.depot_feasible is True 
    assert plan.sequence_feasible is True
    assert plan.added_cost == 798.3

    assert len(plan.trips) == 2

    t1, t2 = plan.trips

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

    assert len(plan.sequence) == 4
    s1, s2, s3, s4 = plan.sequence

    assert s1.trip_id == "2-1"
    assert s1.type == "pickup"
    assert s1.dwell == 180
    assert s1.node.lat == 35.707904816
    assert s1.node.lon == -77.90247345

    assert s2.trip_id == "2-5"
    assert s2.type == "pickup"
    assert s2.dwell == 180
    assert s2.node.lat == 35.720035553
    assert s2.node.lon == -77.893722534

    assert s3.trip_id == "2-1"
    assert s3.type == "dropoff"
    assert s3.dwell == 60
    assert s3.node.lat == 35.737380981
    assert s3.node.lon == -77.906433105

    assert s4.trip_id == "2-5"
    assert s4.type == "dropoff"
    assert s4.dwell == 60
    assert s4.node.lat == 35.753082275
    assert s4.node.lon == -77.930335999

    assert plan.next_immediate_node.lat == 35.723017652422435
    assert plan.next_immediate_node.lon == -77.90871990823223
    assert plan.time_at_next_immediate_node == 17422

    assert plan.veh_travel_time == 178.6
    assert plan.depot_travel_time == 346.5

@pytest.mark.basic
def test_apply_trip_insertion(vehicle, trip_insertion_plan):
    vehicle.trips = {}
    vehicle.stop_sequence = []
    vehicle.time_at_last = 0
    vehicle.time_at_next = 0
    vehicle.last_node = None
    vehicle.rebalancing = True
    
    plan = trip_insertion_plan
    vehicle.apply_trip_insertion(plan)

    assert vehicle.rebalancing is False
    assert vehicle.last_node == plan.next_immediate_node
    assert vehicle.time_at_last == plan.time_at_next_immediate_node
    assert vehicle.time_at_next >= plan.time_at_next_immediate_node + plan.veh_travel_time

    assert set(vehicle.trips.keys()) == {trip.id for trip in plan.trips}
    assert vehicle.stop_sequence == plan.sequence

    first_stop = vehicle.stop_sequence[0]
    first_trip = vehicle.trips[first_stop.trip_id]
    if first_stop.type == first_stop.ACT_PICKUP:
        assert vehicle.time_at_next >= first_trip.pick_up_time
    elif first_stop.type == first_stop.ACT_DROPOFF:
        assert vehicle.time_at_next >= first_trip.earliest_arrival_time



                