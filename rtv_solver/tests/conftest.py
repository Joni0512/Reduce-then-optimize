import pytest

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle, TripInsertionPlan
from rtv_solver.structure.config import Config
from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.assignment_result import AssignmentResult

"""
this is actual data that can be triggered by a single vehicle and requests until 5*3600 + 30*6ß + 5*60
"""

# TODO add basic structures and use them for tests
@pytest.fixture
def single_trip_map():
    return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

@pytest.fixture
def trip_list():
    return [
            Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
            Trip(request_id=2, trip_number=1, am_capacity=1, wc_capacity=0, pick_up_time=19827, latest_pick_up_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
            Trip(request_id=3, trip_number=2, am_capacity=1, wc_capacity=0, pick_up_time=19828, latest_pick_up_time=21628, earliest_arrival_time=20123.7, latest_arrival_time=21923.7, origin=Node(lat=35.725849152, lon=-77.946334839), destination=Node(lat=35.746833801, lon=-77.967590332), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
            Trip(request_id=4, trip_number=3, am_capacity=1, wc_capacity=0, pick_up_time=19851, latest_pick_up_time=21651, earliest_arrival_time=20238.9, latest_arrival_time=22038.9, origin=Node(lat=35.733566284, lon=-77.909339905), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
            Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
            SharedTrip(prev_trip_number=0, trip_number=5, trips={0, 3}, cost=740.8, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)]), 
            SharedTrip(prev_trip_number=1, trip_number=6, trips={1, 2}, cost=766.9, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)]), 
            SharedTrip(prev_trip_number=0, trip_number=7, trips={0, 4}, cost=619.7, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)]), 
            SharedTrip(prev_trip_number=1, trip_number=8, trips={1, 4}, cost=915.0, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)])
        ]

@pytest.fixture
def trip_costs():
    return [
        TripCost(trip_no=0, vehicle_id=0, cost=469.29999999999995, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60)]), 
        TripCost(trip_no=1, vehicle_id=0, cost=790.3, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)]), 
        TripCost(trip_no=2, vehicle_id=0, cost=641.0999999999999, sequence=[VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60)]), 
        TripCost(trip_no=3, vehicle_id=0, cost=526.0, sequence=[VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)]), 
        TripCost(trip_no=4, vehicle_id=0, cost=597.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)]), 
        TripCost(trip_no=5, vehicle_id=0, cost=919.4, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)]), 
        TripCost(trip_no=6, vehicle_id=0, cost=1111.3, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)]), 
        TripCost(trip_no=7, vehicle_id=0, cost=798.3, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)]), 
        TripCost(trip_no=8, vehicle_id=0, cost=1063.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)])
        ]

@pytest.fixture
def vehicle_to_trips_cost_map():
    return {0: [0, 1, 2, 3, 4, 5, 6, 7, 8]}

@pytest.fixture
def trip_to_vehicle_cost_map():
    return {0: [0, 5, 7], 1: [1, 6, 8], 2: [2, 6], 3: [3, 5], 4: [4, 7, 8], 5: [5], 6: [6], 7: [7], 8: [8]}

@pytest.fixture
def request_batch():
    return [
        Request(identifier=1, earliest_pickup_time=19822, latest_pickup_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, am_capacity=1, wc_capacity=0, priority=1), 
        Request(identifier=2, earliest_pickup_time=19827, latest_pickup_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, am_capacity=1, wc_capacity=0, priority=1), 
        Request(identifier=3, earliest_pickup_time=19828, latest_pickup_time=21628, earliest_arrival_time=20123.7, latest_arrival_time=21923.7, origin=Node(lat=35.725849152, lon=-77.946334839), destination=Node(lat=35.746833801, lon=-77.967590332), dwell_pickup=180, dwell_alight=60, am_capacity=1, wc_capacity=0, priority=1), 
        Request(identifier=4, earliest_pickup_time=19851, latest_pickup_time=21651, earliest_arrival_time=20238.9, latest_arrival_time=22038.9, origin=Node(lat=35.733566284, lon=-77.909339905), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, am_capacity=1, wc_capacity=0, priority=1), 
        Request(identifier=5, earliest_pickup_time=20002, latest_pickup_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, am_capacity=1, wc_capacity=0, priority=1)
        ]

@pytest.fixture
def active_requests():
    return {}

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def assignment_result():
    return AssignmentResult(
        vehicle_assignment=
        {
            0: 
                (
                    [ # collection of tripsa
                        Trip( 
                            request_id=1, 
                            trip_number=0, 
                            am_capacity=1, 
                            wc_capacity=0, 
                            pick_up_time=19822, 
                            latest_pick_up_time=21622, 
                            earliest_arrival_time=20112.7, 
                            latest_arrival_time=21912.7, 
                            origin=Node(lat=35.707904816, lon=-77.90247345), 
                            destination=Node(lat=35.737380981, lon=-77.906433105), 
                            dwell_pickup=180, 
                            dwell_alight=60, 
                            iteration=2, ), 
                        Trip(
                            request_id=5, 
                            trip_number=4, 
                            am_capacity=1, 
                            wc_capacity=0, 
                            pick_up_time=20002, 
                            latest_pick_up_time=21802, 
                            earliest_arrival_time=20451.0, 
                            latest_arrival_time=22251.0, 
                            origin=Node(lat=35.720035553, lon=-77.893722534), 
                            destination=Node(lat=35.753082275, lon=-77.930335999), 
                            dwell_pickup=180, 
                            dwell_alight=60, 
                            iteration=2, )
                    ], 
                    [ # sequence
                        VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), 
                        VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), 
                        VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), 
                        VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)
                    ]
                )
            }, 
        request_assignment={1: 0, 5: 0}, 
        rebalancing_assignment={},
        unassigned_trip_count=3, 
        taxi_only_trip_count=2, 
        added_distance=798.3, 
        trip_sizes=[2], 
        status=2, 
        runtime=0.0037648677825927734)

@pytest.fixture
def vehicle():
    return Vehicle(vehicle_id=0, start_node=Node(lat=35.723017652422435, lon=-77.90871990823223), am_capacity=8, wc_capacity=3, start_time=18000, end_time=72000, depot=Node(lat=35.723017652422435, lon=-77.90871990823223))

@pytest.fixture
def trip_insertion_plan():
    return TripInsertionPlan(
        depot_feasible=True, 
        sequence_feasible=True, 
        added_cost=798.3, 
        sequence=[
            VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)], 
        trips=[
            Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
            Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, )
            ], 
        next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), 
        time_at_next_immediate_node=17422, 
        veh_travel_time=178.6, 
        depot_travel_time=346.5)