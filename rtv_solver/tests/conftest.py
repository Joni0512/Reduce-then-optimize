import pytest
import numpy as np
from pathlib import Path
import requests

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.trip_insertion_plan import TripInsertionPlan
from rtv_solver.structure.config import Config
from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.assignment_result import AssignmentResult
from rtv_solver.structure.driver_run import ManifestEntry

"""
this is actual data that can be triggered by a single vehicle and requests until 5*3600 + 30*60+ 5*60

Interim trip costs and the related test objects are checking whether also a misordering of items works correctly. This has raised issues and thus the test was integrated.

If a fixture is created, sometimes also other objects have large changes that mirror these changes.
"""

SERVER_URL = "http://127.0.0.1:5001/"


def is_server_running():
    try:
        requests.get(SERVER_URL, timeout=1)
        return True
    except requests.ConnectionError:
        return False

def pytest_collection_modifyitems(config, items):
    """pytest setup with hook to pytest-config and all test-items to skip tests that require the server to be running"""
    if is_server_running():
        return

    skip_server = pytest.mark.skip(reason="Server not running")

    for item in items:
        if "server" in item.keywords:
            item.add_marker(skip_server)

# TODO add basic structures and use them for tests
@pytest.fixture
def single_trip_map_neworder():
    return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

@pytest.fixture
def vehicle_to_trips_cost_map_neworder():
    return {0: [0, 1, 2, 3, 4, 5, 6, 7, 8]}

@pytest.fixture
def trip_to_vehicle_cost_map_neworder():
    return  {0: [0, 5, 8], 1: [1, 6, 7], 2: [4, 6], 3: [2, 5], 4: [3, 7, 8], 5: [5], 6: [8], 7: [6], 8: [7]}
    
@pytest.fixture
def trip_list_neworder():
    return [
        Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
        Trip(request_id=2, trip_number=1, am_capacity=1, wc_capacity=0, pick_up_time=19827, latest_pick_up_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
        Trip(request_id=3, trip_number=2, am_capacity=1, wc_capacity=0, pick_up_time=19828, latest_pick_up_time=21628, earliest_arrival_time=20123.7, latest_arrival_time=21923.7, origin=Node(lat=35.725849152, lon=-77.946334839), destination=Node(lat=35.746833801, lon=-77.967590332), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
        Trip(request_id=4, trip_number=3, am_capacity=1, wc_capacity=0, pick_up_time=19851, latest_pick_up_time=21651, earliest_arrival_time=20238.9, latest_arrival_time=22038.9, origin=Node(lat=35.733566284, lon=-77.909339905), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
        Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, ), 
        SharedTrip(prev_trip_number=0, trip_number=5, trips={0, 3}, cost=740.8, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)]), 
        SharedTrip(prev_trip_number=0, trip_number=6, trips={0, 4}, cost=619.7, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)]), 
        SharedTrip(prev_trip_number=1, trip_number=7, trips={1, 2}, cost=766.9, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)]), 
        SharedTrip(prev_trip_number=1, trip_number=8, trips={1, 4}, cost=915.0, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)])
        ]    

@pytest.fixture
def trip_costs_neworder():
    return [
        TripCost(trip_no=0, vehicle_id=0, cost=469.29999999999995, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=469.29999999999995, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60)], trips=[Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=178.6, depot_travel_time=206.4, direct_trip_times=[290.7], total_direct_travel_time=290.7, actual_travel_time=469.29999999999995, total_dwell_time=240.0, actual_route_travel_time=709.3, detour_time=178.59999999999997, idling_time=721.4)), 
        TripCost(trip_no=1, vehicle_id=0, cost=790.3, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=790.3, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], trips=[Trip(request_id=2, trip_number=1, am_capacity=1, wc_capacity=0, pick_up_time=19827, latest_pick_up_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=344.4, depot_travel_time=463.2, direct_trip_times=[445.9], total_direct_travel_time=445.9, actual_travel_time=790.3, total_dwell_time=240.0, actual_route_travel_time=1030.3, detour_time=344.4, idling_time=560.6)), 
        TripCost(trip_no=3, vehicle_id=0, cost=526.0, sequence=[VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=526.0, sequence=[VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)], trips=[Trip(request_id=4, trip_number=3, am_capacity=1, wc_capacity=0, pick_up_time=19851, latest_pick_up_time=21651, earliest_arrival_time=20238.9, latest_arrival_time=22038.9, origin=Node(lat=35.733566284, lon=-77.909339905), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=138.1, depot_travel_time=266.8, direct_trip_times=[387.9], total_direct_travel_time=387.9, actual_travel_time=526.0, total_dwell_time=240.0, actual_route_travel_time=766.0, detour_time=138.10000000000002, idling_time=790.9)), 
        TripCost(trip_no=4, vehicle_id=0, cost=597.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=597.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)], trips=[Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=148.1, depot_travel_time=346.5, direct_trip_times=[449.0], total_direct_travel_time=449.0, actual_travel_time=597.1, total_dwell_time=240.0, actual_route_travel_time=837.1, detour_time=148.10000000000002, idling_time=931.9)), 
        TripCost(trip_no=2, vehicle_id=0, cost=641.0999999999999, sequence=[VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=641.0999999999999, sequence=[VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60)], trips=[Trip(request_id=3, trip_number=2, am_capacity=1, wc_capacity=0, pick_up_time=19828, latest_pick_up_time=21628, earliest_arrival_time=20123.7, latest_arrival_time=21923.7, origin=Node(lat=35.725849152, lon=-77.946334839), destination=Node(lat=35.746833801, lon=-77.967590332), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=341.4, depot_travel_time=496.4, direct_trip_times=[299.7], total_direct_travel_time=299.7, actual_travel_time=641.0999999999999, total_dwell_time=240.0, actual_route_travel_time=881.0999999999999, detour_time=341.3999999999999, idling_time=564.6)), 
        TripCost(trip_no=5, vehicle_id=0, cost=919.4, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=919.4, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-4', node=Node(lat=35.733566284, lon=-77.909339905), type='pickup', dwell=180), VehicleStop(trip_id='2-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)], trips=[Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, ), Trip(request_id=4, trip_number=3, am_capacity=1, wc_capacity=0, pick_up_time=19851, latest_pick_up_time=21651, earliest_arrival_time=20238.9, latest_arrival_time=22038.9, origin=Node(lat=35.733566284, lon=-77.909339905), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=178.6, depot_travel_time=266.8, direct_trip_times=[290.7, 387.9], total_direct_travel_time=678.5999999999999, actual_travel_time=919.4, total_dwell_time=480.0, actual_route_travel_time=1399.4, detour_time=240.80000000000007, idling_time=721.4)), 
        TripCost(trip_no=7, vehicle_id=0, cost=1111.3, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=1111.3, sequence=[VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.725849152, lon=-77.946334839), type='pickup', dwell=180), VehicleStop(trip_id='2-3', node=Node(lat=35.746833801, lon=-77.967590332), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], trips=[Trip(request_id=2, trip_number=1, am_capacity=1, wc_capacity=0, pick_up_time=19827, latest_pick_up_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, iteration=2, ), Trip(request_id=3, trip_number=2, am_capacity=1, wc_capacity=0, pick_up_time=19828, latest_pick_up_time=21628, earliest_arrival_time=20123.7, latest_arrival_time=21923.7, origin=Node(lat=35.725849152, lon=-77.946334839), destination=Node(lat=35.746833801, lon=-77.967590332), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=344.4, depot_travel_time=463.2, direct_trip_times=[445.9, 299.7], total_direct_travel_time=745.5999999999999, actual_travel_time=1111.3, total_dwell_time=480.0, actual_route_travel_time=1591.3, detour_time=365.70000000000005, idling_time=560.6)), 
        TripCost(trip_no=8, vehicle_id=0, cost=1063.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=1063.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], trips=[Trip(request_id=2, trip_number=1, am_capacity=1, wc_capacity=0, pick_up_time=19827, latest_pick_up_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, iteration=2, ), Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=148.1, depot_travel_time=463.2, direct_trip_times=[445.9, 449.0], total_direct_travel_time=894.9, actual_travel_time=1063.1, total_dwell_time=480.0, actual_route_travel_time=1543.1, detour_time=168.19999999999993, idling_time=756.9)), 
        TripCost(trip_no=6, vehicle_id=0, cost=798.3, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=798.3, sequence=[VehicleStop(trip_id='2-1', node=Node(lat=35.707904816, lon=-77.90247345), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-1', node=Node(lat=35.737380981, lon=-77.906433105), type='dropoff', dwell=60), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60)], trips=[Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, ), Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=178.6, depot_travel_time=346.5, direct_trip_times=[290.7, 449.0], total_direct_travel_time=739.7, actual_travel_time=798.3, total_dwell_time=480.0, actual_route_travel_time=1278.3, detour_time=58.59999999999991, idling_time=721.4))
        ]

@pytest.fixture 
def trip():
    return Trip(request_id=1, trip_number=0, am_capacity=1, wc_capacity=0, pick_up_time=19822, latest_pick_up_time=21622, earliest_arrival_time=20112.7, latest_arrival_time=21912.7, origin=Node(lat=35.707904816, lon=-77.90247345), destination=Node(lat=35.737380981, lon=-77.906433105), dwell_pickup=180, dwell_alight=60, iteration=2, )

@pytest.fixture
def single_trip_map():
    return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

@pytest.fixture
def vehicle_to_trips_cost_map():
    return {0: [0, 1, 2, 3, 4, 5, 6, 7, 8]}

@pytest.fixture
def trip_to_vehicle_cost_map():
    return {
        0: [0, 5, 7], 
        1: [1, 6, 8], 
        2: [2, 6], 
        3: [3, 5], 
        4: [4, 7, 8], 
        5: [5], 
        6: [6], 
        7: [7], 
        8: [8]
        }

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
    """define a config that is always stable and can be used in all tests"""
    return Config(
    CONFIG_FILE="config.yaml",
    OVERRIDE=[],
    OUTPUT_DIR=Path("outputs") / "debug",
    INPUT_FILE="rtv-solver/inputs/wilson_nc_initial.pkl",
    SERVER_URL="http://127.0.0.1:5001/",
    MAX_THREAD_CNT=16,
    RTV_TIMEOUT=120,
    ILP_TIMEOUT=120,
    ILP_PENALTY=1_000_000,
    MODE="offline",
    MAX_CARDINALITY=2,
    LARGEST_TSP=8,
    SHARE_COST_FACTOR=10.0,
    REBALANCING=False,
    KEEP_ACTIVE=True,
    RETURN_DEPOT=False,
    WALK_DISTANCE_CUTOFF=0,
    STEP_SIZE=300,
    BATCH_INTERVAL=1200,
    DWELL_PICKUP=180,
    DWELL_ALIGHT=60,
    TRAVEL_TIME_MARGIN=5,
    )

@pytest.fixture
def assignment_result():
    return AssignmentResult(
        vehicle_assignment=
        {
            0: 
                (
                    [ # collection of trips
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
def vehicle_init():
    return Vehicle(vehicle_id=0, start_node=Node(lat=35.723017652422435, lon=-77.90871990823223), am_capacity=8, wc_capacity=3, start_time=18000, end_time=72000, depot=Node(lat=35.723017652422435, lon=-77.90871990823223), manifest=[])

@pytest.fixture
def vehicle_0(vehicles_interim):
    return vehicles_interim[0]

@pytest.fixture
def vehicle_1(vehicles_interim):
    return vehicles_interim[1]

@pytest.fixture
def vehicle_2(vehicles_interim):
    return vehicles_interim[2]
    
@pytest.fixture
def vehicle_3(vehicles_interim):
    return vehicles_interim[3]

@pytest.fixture
def vehicles_start() -> dict[int, Vehicle]:
    return {
        0: Vehicle.from_dict( 
            {'id': 0, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 8, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 17422, 'last_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'trips': {}, 'picked': [], 'served_trips': [], 'stop_sequence': [], 'final_stop_time': 18000, 'manifest': []}
        ),
        1: Vehicle.from_dict(
            {'id': 1, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 8, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 17422, 'last_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'trips': {}, 'picked': [], 'served_trips': [], 'stop_sequence': [], 'final_stop_time': 18000, 'manifest': []}
        ),
        2: Vehicle.from_dict(
            {'id': 2, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 8, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 17422, 'last_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'trips': {}, 'picked': [], 'served_trips': [], 'stop_sequence': [], 'final_stop_time': 18000, 'manifest': []}
        ),
        3: Vehicle.from_dict(
            {'id': 3, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 8, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 17422, 'last_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.723017652422435, lon=-77.90871990823223), 'trips': {}, 'picked': [], 'served_trips': [], 'stop_sequence': [], 'final_stop_time': 18000, 'manifest': []}
        )
    }

@pytest.fixture
def vehicles_interim() -> dict[int, Vehicle]:
    return {
        0: Vehicle.from_dict(
            {'id': 0, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 8, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 21640.8, 'last_node': Node(lat=35.740951538, lon=-77.963066101), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.740951538, lon=-77.963066101), 'trips': {}, 'picked': [], 'served_trips': [], 'stop_sequence': [], 'final_stop_time': 18000, 'manifest': [ManifestEntry(run_id=0, booking_id=2, order=1, action='pickup', loc=Node(lat=35.75359726, lon=-77.924423218), am=1, wc=0, scheduled_time=19827, time_window_start=19827, time_window_end=21627), ManifestEntry(run_id=0, booking_id=8, order=2, action='pickup', loc=Node(lat=35.731918335, lon=-77.920219421), am=1, wc=0, scheduled_time=20423, time_window_start=20423, time_window_end=22223), ManifestEntry(run_id=0, booking_id=3, order=3, action='pickup', loc=Node(lat=35.725849152, lon=-77.946334839), am=1, wc=0, scheduled_time=20869.0, time_window_start=19828, time_window_end=21628), ManifestEntry(run_id=0, booking_id=3, order=4, action='dropoff', loc=Node(lat=35.746833801, lon=-77.967590332), am=1, wc=0, scheduled_time=21348.7, time_window_start=20123.7, time_window_end=21923.7), ManifestEntry(run_id=0, booking_id=8, order=5, action='dropoff', loc=Node(lat=35.746833801, lon=-77.967590332), am=1, wc=0, scheduled_time=21408.7, time_window_start=20807.1, time_window_end=22607.1), ManifestEntry(run_id=0, booking_id=2, order=6, action='dropoff', loc=Node(lat=35.740951538, lon=-77.963066101), am=1, wc=0, scheduled_time=21580.8, time_window_start=20270.9, time_window_end=22070.9)]}
        ),
        1: Vehicle.from_dict(
            {'id': 1, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 7, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 21124.4, 'last_node': Node(lat=35.714965, lon=-77.952164), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.714965, lon=-77.952164), 'trips': {'3-12': Trip(request_id=12, trip_number=-5, am_capacity=1, wc_capacity=0, pick_up_time=20708, latest_pick_up_time=22508, earliest_arrival_time=21033.7, latest_arrival_time=22833.7, origin=Node(lat=35.721488953, lon=-77.923782349), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, iteration=3, )}, 'picked': ['3-12'], 'served_trips': [], 'stop_sequence': [VehicleStop(trip_id='3-12', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)], 'final_stop_time': 18000, 'manifest': [ManifestEntry(run_id=1, booking_id=12, order=1, action='pickup', loc=Node(lat=35.721488953, lon=-77.923782349), am=1, wc=0, scheduled_time=20708, time_window_start=20708, time_window_end=22508), ManifestEntry(run_id=1, booking_id=17, order=2, action='pickup', loc=Node(lat=35.71496582, lon=-77.952163696), am=1, wc=0, scheduled_time=21718, time_window_start=21718, time_window_end=23518), ManifestEntry(run_id=1, booking_id=14, order=3, action='pickup', loc=Node(lat=35.700576782, lon=-77.928794861), am=1, wc=0, scheduled_time=22236.6, time_window_start=20931, time_window_end=22731), ManifestEntry(run_id=1, booking_id=14, order=4, action='dropoff', loc=Node(lat=35.700027466, lon=-77.902740479), am=1, wc=0, scheduled_time=22708.399999999998, time_window_start=21222.8, time_window_end=23022.8), ManifestEntry(run_id=1, booking_id=17, order=5, action='dropoff', loc=Node(lat=35.700027466, lon=-77.902740479), am=1, wc=0, scheduled_time=22768.399999999998, time_window_start=22186.8, time_window_end=23986.8), ManifestEntry(run_id=1, booking_id=12, order=6, action='dropoff', loc=Node(lat=35.699867249, lon=-77.902770996), am=1, wc=0, scheduled_time=22832.1, time_window_start=21033.7, time_window_end=22833.7)]}
        ),
        2: Vehicle.from_dict(
            {'id': 2, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 6, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 21637.299999999996, 'last_node': Node(lat=35.720367, lon=-77.905554), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.720367, lon=-77.905554), 'trips': {'3-13': Trip(request_id=13, trip_number=-6, am_capacity=1, wc_capacity=0, pick_up_time=20884, latest_pick_up_time=22684, earliest_arrival_time=21170.3, latest_arrival_time=22970.3, origin=Node(lat=35.737335205, lon=-77.906143188), destination=Node(lat=35.707904816, lon=-77.90247345), dwell_pickup=180, dwell_alight=60, iteration=3, ), '3-4': Trip(request_id=4, trip_number=-1, am_capacity=1, wc_capacity=0, pick_up_time=19851, latest_pick_up_time=21651, earliest_arrival_time=20238.9, latest_arrival_time=22038.9, origin=Node(lat=35.733566284, lon=-77.909339905), destination=Node(lat=35.699867249, lon=-77.902770996), dwell_pickup=180, dwell_alight=60, iteration=3, )}, 'picked': ['3-13', '3-4'], 'served_trips': [], 'stop_sequence': [VehicleStop(trip_id='3-13', node=Node(lat=35.707904816, lon=-77.90247345), type='dropoff', dwell=60), VehicleStop(trip_id='3-4', node=Node(lat=35.699867249, lon=-77.902770996), type='dropoff', dwell=60)], 'final_stop_time': 18000, 'manifest': [ManifestEntry(run_id=2, booking_id=1, order=1, action='pickup', loc=Node(lat=35.707904816, lon=-77.90247345), am=1, wc=0, scheduled_time=19822, time_window_start=19822, time_window_end=21622), ManifestEntry(run_id=2, booking_id=7, order=2, action='pickup', loc=Node(lat=35.707904816, lon=-77.900428772), am=1, wc=0, scheduled_time=20385, time_window_start=20385, time_window_end=22185), ManifestEntry(run_id=2, booking_id=13, order=3, action='pickup', loc=Node(lat=35.737335205, lon=-77.906143188), am=1, wc=0, scheduled_time=20884, time_window_start=20884, time_window_end=22684), ManifestEntry(run_id=2, booking_id=1, order=4, action='dropoff', loc=Node(lat=35.737380981, lon=-77.906433105), am=1, wc=0, scheduled_time=21065.6, time_window_start=20112.7, time_window_end=21912.7), ManifestEntry(run_id=2, booking_id=4, order=5, action='pickup', loc=Node(lat=35.733566284, lon=-77.909339905), am=1, wc=0, scheduled_time=21187.8, time_window_start=19851, time_window_end=21651), ManifestEntry(run_id=2, booking_id=7, order=6, action='dropoff', loc=Node(lat=35.733039856, lon=-77.914161682), am=1, wc=0, scheduled_time=21418.1, time_window_start=20705.8, time_window_end=22505.8), ManifestEntry(run_id=2, booking_id=13, order=7, action='dropoff', loc=Node(lat=35.707904816, lon=-77.90247345), am=1, wc=0, scheduled_time=21775.3, time_window_start=21170.3, time_window_end=22970.3), ManifestEntry(run_id=2, booking_id=4, order=8, action='dropoff', loc=Node(lat=35.699867249, lon=-77.902770996), am=1, wc=0, scheduled_time=21996.399999999998, time_window_start=20238.9, time_window_end=22038.9)]}
        ),
        3:  Vehicle.from_dict(
            {'id': 3, 'start_time': 18000, 'end_time': 72000, 'depot': Node(lat=35.723017652422435, lon=-77.90871990823223), 'am_capacity': 5, 'wc_capacity': 3, 'started': True, 'rebalancing': False, 'dwelling': False, 'time_at_last': 21646.199999999997, 'last_node': Node(lat=35.753192, lon=-77.930922), 'time_at_next': 18000, 'next_immediate_node': Node(lat=35.753192, lon=-77.930922), 'trips': {'3-5': Trip(request_id=5, trip_number=-2, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=3, ), '3-6': Trip(request_id=6, trip_number=-3, am_capacity=1, wc_capacity=0, pick_up_time=20139, latest_pick_up_time=21939, earliest_arrival_time=20457.8, latest_arrival_time=22257.8, origin=Node(lat=35.728252411, lon=-77.909576416), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=3, ), '3-10': Trip(request_id=10, trip_number=-4, am_capacity=1, wc_capacity=0, pick_up_time=20567, latest_pick_up_time=22367, earliest_arrival_time=20978.3, latest_arrival_time=22778.3, origin=Node(lat=35.721466064, lon=-77.900192261), destination=Node(lat=35.737010956, lon=-77.944717407), dwell_pickup=180, dwell_alight=60, iteration=3, )}, 'picked': ['3-5', '3-6', '3-10'], 'served_trips': [], 'stop_sequence': [VehicleStop(trip_id='3-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='3-6', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='3-10', node=Node(lat=35.737010956, lon=-77.944717407), type='dropoff', dwell=60)], 'final_stop_time': 18000, 'manifest': [ManifestEntry(run_id=3, booking_id=10, order=1, action='pickup', loc=Node(lat=35.721466064, lon=-77.900192261), am=1, wc=0, scheduled_time=20567, time_window_start=20567, time_window_end=22367), ManifestEntry(run_id=3, booking_id=5, order=2, action='pickup', loc=Node(lat=35.720035553, lon=-77.893722534), am=1, wc=0, scheduled_time=20824.6, time_window_start=20002, time_window_end=21802), ManifestEntry(run_id=3, booking_id=6, order=3, action='pickup', loc=Node(lat=35.728252411, lon=-77.909576416), am=1, wc=0, scheduled_time=21166.1, time_window_start=20139, time_window_end=21939), ManifestEntry(run_id=3, booking_id=5, order=4, action='dropoff', loc=Node(lat=35.753082275, lon=-77.930335999), am=1, wc=0, scheduled_time=21664.899999999998, time_window_start=20451.0, time_window_end=22251.0), ManifestEntry(run_id=3, booking_id=6, order=5, action='dropoff', loc=Node(lat=35.753082275, lon=-77.930335999), am=1, wc=0, scheduled_time=21724.899999999998, time_window_start=20457.8, time_window_end=22257.8), ManifestEntry(run_id=3, booking_id=10, order=6, action='dropoff', loc=Node(lat=35.737010956, lon=-77.944717407), am=1, wc=0, scheduled_time=21994.6, time_window_start=20978.3, time_window_end=22778.3)]}
        )
    }
        
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

@pytest.fixture
def payload():
    """basic payload with limited requests"""
    return {
        'depot': {'pt': {'lat': 35.723017652422435, 'lon': -77.90871990823223}, 'node_id': 0}, 
        'requests': [{'booking_id': np.float64(1.0), 'pickup_pt': {'lon': np.float64(-77.90247345), 'lat': np.float64(35.707904816)}, 'dropoff_pt': {'lon': np.float64(-77.906433105), 'lat': np.float64(35.737380981)}, 'pickup_time_window_start': 19822, 'pickup_time_window_end': 21622, 'dropoff_time_window_start': 20112.7, 'dropoff_time_window_end': 21912.7, 'am': 1, 'wc': 0}, {'booking_id': np.float64(2.0), 'pickup_pt': {'lon': np.float64(-77.924423218), 'lat': np.float64(35.75359726)}, 'dropoff_pt': {'lon': np.float64(-77.963066101), 'lat': np.float64(35.740951538)}, 'pickup_time_window_start': 19827, 'pickup_time_window_end': 21627, 'dropoff_time_window_start': 20270.9, 'dropoff_time_window_end': 22070.9, 'am': 1, 'wc': 0}, {'booking_id': np.float64(3.0), 'pickup_pt': {'lon': np.float64(-77.946334839), 'lat': np.float64(35.725849152)}, 'dropoff_pt': {'lon': np.float64(-77.967590332), 'lat': np.float64(35.746833801)}, 'pickup_time_window_start': 19828, 'pickup_time_window_end': 21628, 'dropoff_time_window_start': 20123.7, 'dropoff_time_window_end': 21923.7, 'am': 1, 'wc': 0}, {'booking_id': np.float64(4.0), 'pickup_pt': {'lon': np.float64(-77.909339905), 'lat': np.float64(35.733566284)}, 'dropoff_pt': {'lon': np.float64(-77.902770996), 'lat': np.float64(35.699867249)}, 'pickup_time_window_start': 19851, 'pickup_time_window_end': 21651, 'dropoff_time_window_start': 20238.9, 'dropoff_time_window_end': 22038.9, 'am': 1, 'wc': 0}, {'booking_id': np.float64(5.0), 'pickup_pt': {'lon': np.float64(-77.893722534), 'lat': np.float64(35.720035553)}, 'dropoff_pt': {'lon': np.float64(-77.930335999), 'lat': np.float64(35.753082275)}, 'pickup_time_window_start': 20002, 'pickup_time_window_end': 21802, 'dropoff_time_window_start': 20451, 'dropoff_time_window_end': 22251, 'am': 1, 'wc': 0}, {'booking_id': np.float64(6.0), 'pickup_pt': {'lon': np.float64(-77.909576416), 'lat': np.float64(35.728252411)}, 'dropoff_pt': {'lon': np.float64(-77.930335999), 'lat': np.float64(35.753082275)}, 'pickup_time_window_start': 20139, 'pickup_time_window_end': 21939, 'dropoff_time_window_start': 20457.8, 'dropoff_time_window_end': 22257.8, 'am': 1, 'wc': 0}, {'booking_id': np.float64(7.0), 'pickup_pt': {'lon': np.float64(-77.900428772), 'lat': np.float64(35.707904816)}, 'dropoff_pt': {'lon': np.float64(-77.914161682), 'lat': np.float64(35.733039856)}, 'pickup_time_window_start': 20385, 'pickup_time_window_end': 22185, 'dropoff_time_window_start': 20705.8, 'dropoff_time_window_end': 22505.8, 'am': 1, 'wc': 0}, {'booking_id': np.float64(8.0), 'pickup_pt': {'lon': np.float64(-77.920219421), 'lat': np.float64(35.731918335)}, 'dropoff_pt': {'lon': np.float64(-77.967590332), 'lat': np.float64(35.746833801)}, 'pickup_time_window_start': 20423, 'pickup_time_window_end': 22223, 'dropoff_time_window_start': 20807.1, 'dropoff_time_window_end': 22607.1, 'am': 1, 'wc': 0}, {'booking_id': np.float64(9.0), 'pickup_pt': {'lon': np.float64(-77.944015503), 'lat': np.float64(35.749149323)}, 'dropoff_pt': {'lon': np.float64(-77.939537048), 'lat': np.float64(35.768268585)}, 'pickup_time_window_start': 20463, 'pickup_time_window_end': 22263, 'dropoff_time_window_start': 20713.5, 'dropoff_time_window_end': 22513.5, 'am': 1, 'wc': 0}, {'booking_id': np.float64(10.0), 'pickup_pt': {'lon': np.float64(-77.900192261), 'lat': np.float64(35.721466064)}, 'dropoff_pt': {'lon': np.float64(-77.944717407), 'lat': np.float64(35.737010956)}, 'pickup_time_window_start': 20567, 'pickup_time_window_end': 22367, 'dropoff_time_window_start': 20978.3, 'dropoff_time_window_end': 22778.3, 'am': 1, 'wc': 0}, {'booking_id': np.float64(11.0), 'pickup_pt': {'lon': np.float64(-77.887863159), 'lat': np.float64(35.716983795)}, 'dropoff_pt': {'lon': np.float64(-77.902740479), 'lat': np.float64(35.700027466)}, 'pickup_time_window_start': 20625, 'pickup_time_window_end': 22425, 'dropoff_time_window_start': 20965.7, 'dropoff_time_window_end': 22765.7, 'am': 1, 'wc': 0}, {'booking_id': np.float64(12.0), 'pickup_pt': {'lon': np.float64(-77.923782349), 'lat': np.float64(35.721488953)}, 'dropoff_pt': {'lon': np.float64(-77.902770996), 'lat': np.float64(35.699867249)}, 'pickup_time_window_start': 20708, 'pickup_time_window_end': 22508, 'dropoff_time_window_start': 21033.7, 'dropoff_time_window_end': 22833.7, 'am': 1, 'wc': 0}, {'booking_id': np.float64(13.0), 'pickup_pt': {'lon': np.float64(-77.906143188), 'lat': np.float64(35.737335205)}, 'dropoff_pt': {'lon': np.float64(-77.90247345), 'lat': np.float64(35.707904816)}, 'pickup_time_window_start': 20884, 'pickup_time_window_end': 22684, 'dropoff_time_window_start': 21170.3, 'dropoff_time_window_end': 22970.3, 'am': 1, 'wc': 0}, {'booking_id': np.float64(14.0), 'pickup_pt': {'lon': np.float64(-77.928794861), 'lat': np.float64(35.700576782)}, 'dropoff_pt': {'lon': np.float64(-77.902740479), 'lat': np.float64(35.700027466)}, 'pickup_time_window_start': 20931, 'pickup_time_window_end': 22731, 'dropoff_time_window_start': 21222.8, 'dropoff_time_window_end': 23022.8, 'am': 1, 'wc': 0}, {'booking_id': np.float64(15.0), 'pickup_pt': {'lon': np.float64(-77.919136047), 'lat': np.float64(35.69877243)}, 'dropoff_pt': {'lon': np.float64(-77.913574219), 'lat': np.float64(35.740600586)}, 'pickup_time_window_start': 21481, 'pickup_time_window_end': 23281, 'dropoff_time_window_start': 22049.1, 'dropoff_time_window_end': 23849.1, 'am': 1, 'wc': 0}, {'booking_id': np.float64(16.0), 'pickup_pt': {'lon': np.float64(-77.939537048), 'lat': np.float64(35.768268585)}, 'dropoff_pt': {'lon': np.float64(-77.942878723), 'lat': np.float64(35.748065948)}, 'pickup_time_window_start': 21585, 'pickup_time_window_end': 23385, 'dropoff_time_window_start': 21831.9, 'dropoff_time_window_end': 23631.9, 'am': 1, 'wc': 0}], 
        'driver_runs': [
            {'state': {'run_id': 0, 'start_time': 18000, 'end_time': 72000, 'am_capacity': 8, 'wc_capacity': 3, 'locations_already_serviced': 0, 'location_dt_seconds': 0, 'loc': {'lat': 35.723017652422435, 'lon': -77.90871990823223}}, 'manifest': []}, 
            {'state': {'run_id': 1, 'start_time': 18000, 'end_time': 72000, 'am_capacity': 8, 'wc_capacity': 3, 'locations_already_serviced': 0, 'location_dt_seconds': 0, 'loc': {'lat': 35.723017652422435, 'lon': -77.90871990823223}}, 'manifest': []}, 
            {'state': {'run_id': 2, 'start_time': 18000, 'end_time': 72000, 'am_capacity': 8, 'wc_capacity': 3, 'locations_already_serviced': 0, 'location_dt_seconds': 0, 'loc': {'lat': 35.723017652422435, 'lon': -77.90871990823223}}, 'manifest': []}, 
            {'state': {'run_id': 3, 'start_time': 18000, 'end_time': 72000, 'am_capacity': 8, 'wc_capacity': 3, 'locations_already_serviced': 0, 'location_dt_seconds': 0, 'loc': {'lat': 35.723017652422435, 'lon': -77.90871990823223}}, 'manifest': []}
            ]
        }