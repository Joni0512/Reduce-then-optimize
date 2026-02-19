import pytest

from rtv_solver.pipeline import FeatureBuilder
from rtv_solver.pipeline.feat_builder import VehicleFeatures, StateFeatures
from rtv_solver.pipeline.feat_builder import TripCostFeatures

from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.config import Config

@pytest.mark.basic
def test_vehicle_features_start(config: Config, payload: dict, vehicles_start: dict[int, Vehicle]): 
    """test before the vehicles have even really started"""
    fb = FeatureBuilder(payload, config)

    current_time = 18000
    vehicle = vehicles_start[0]

    feat_dict = fb._vehicle_features(vehicle, vehicles_start, current_time)
    
    vehicle_features = VehicleFeatures(**feat_dict)
    # simplified to turn dict into a vehicle_features-object to improve assert
    assert vehicle_features.v_norm_lat_next_position == pytest.approx(0.3488701074713566)
    assert vehicle_features.v_norm_lon_next_position == pytest.approx(0.7388165129920522)
    assert vehicle_features.v_norm_interval_remaining_boarded_time == 0.0
    assert vehicle_features.v_norm_remaining_operating_period == 1.0
    assert vehicle_features.v_norm_step_remaining_boarded_time == 0.0
    assert vehicle_features.v_norm_remaining_am_cap == 1.0
    assert vehicle_features.v_norm_remaining_wc_cap == 1.0
    assert vehicle_features.v_norm_vehicle_count_in_proximity == 1.0
    assert vehicle_features.v_avg_vehicle_distance == 0.0

@pytest.mark.basic
def test_vehicle_features_interim(config: Config, payload: dict, vehicles_interim: dict[int, Vehicle]):
    """test when vehicles are already underway"""
    fb = FeatureBuilder(payload, config)

    current_time = 20000
    vehicle = vehicles_interim[2]

    feat_dict = fb._vehicle_features(vehicle, vehicles_interim, current_time)
    
    vehicle_features = VehicleFeatures(**feat_dict)
    # simplified to turn dict into a vehicle_features-object to improve assert
    assert vehicle_features.v_norm_lat_next_position == pytest.approx(0.31072918874002)
    assert vehicle_features.v_norm_lon_next_position == pytest.approx(0.7785740550425999)
    assert vehicle_features.v_norm_interval_remaining_boarded_time == pytest.approx((21996.399999999998 - current_time) / config.BATCH_INTERVAL)
    assert vehicle_features.v_norm_remaining_operating_period == pytest.approx(0.658067053)
    assert vehicle_features.v_norm_step_remaining_boarded_time == pytest.approx((21996.39999999998 - current_time) / config.STEP_SIZE)
    assert vehicle_features.v_norm_remaining_am_cap == 0.75
    assert vehicle_features.v_norm_remaining_wc_cap == 1.0
    assert vehicle_features.v_norm_vehicle_count_in_proximity == 0.0
    assert vehicle_features.v_avg_vehicle_distance == pytest.approx(0.8995631130)

@pytest.mark.basic
def test_state_features_start(config: Config, payload, vehicles_start):
    fb = FeatureBuilder(payload, config)

    current_time = 18000

    feat_dict = fb._state_features(current_time, vehicles_start)

    state_features = StateFeatures(**feat_dict)

    assert state_features.s_norm_time == 0.0
    assert state_features.s_avg_remaining_am_cap == 1.0
    assert state_features.s_avg_remaining_wc_cap == 1.0
    assert state_features.s_avg_remaining_interval_boarded_time == 0.0
    assert state_features.s_avg_remaining_step_boarded_time == 0.0
    assert state_features.s_total_vehicles == 4

@pytest.mark.basic
def test_state_features_interim(config: Config, payload, vehicles_interim):
    fb = FeatureBuilder(payload, config)

    current_time = 18000

    feat_dict = fb._state_features(current_time, vehicles_interim)

    state_features = StateFeatures(**feat_dict)

    assert state_features.s_norm_time == 0.0
    assert state_features.s_avg_remaining_am_cap == 0.8125
    assert state_features.s_avg_remaining_wc_cap == 1.0
    assert state_features.s_total_vehicles == 4
    # NOTE below could fail if one changes defaults in the config, as it was easier this way
    remaining_interval_boarded_time = 2.6714791666666655
    assert state_features.s_avg_remaining_interval_boarded_time == pytest.approx(remaining_interval_boarded_time)
    assert state_features.s_avg_remaining_step_boarded_time == pytest.approx( remaining_interval_boarded_time * config.BATCH_INTERVAL / config.STEP_SIZE)

@pytest.mark.basic
def test_trip_cost_features_single(config: Config, payload: dict, trip_costs_neworder: list[TripCost]):
    fb = FeatureBuilder(payload, config)

    current_time = 20000
    trip_cost = trip_costs_neworder[0]

    feat_dict = fb._trip_cost_features(trip_cost, current_time)
    trip_cost_features = TripCostFeatures(**feat_dict)

    assert trip_cost_features.tc_cost == pytest.approx(469.29999999999995)
    assert trip_cost_features.tc_sequence_len == 2
    assert trip_cost_features.tc_num_trips == 1
    assert trip_cost_features.tc_total_am_demand == 1
    assert trip_cost_features.tc_total_wc_demand == 0

    assert trip_cost_features.tc_norm_travel_time_to_first_pickup == pytest.approx(0.01691548)
    assert trip_cost_features.tc_total_direct_travel_time == pytest.approx(290.7)
    assert trip_cost_features.tc_actual_travel_time == pytest.approx(469.29999999999995)
    assert trip_cost_features.tc_norm_batch_actual_travel_time == pytest.approx(0.39108333333333)
    assert trip_cost_features.tc_total_dwell_time == pytest.approx(240.0)
    assert trip_cost_features.tc_dwell_time_ratio == pytest.approx(0.51139996)
    assert trip_cost_features.tc_detour_time == pytest.approx(178.59999999999997)
    assert trip_cost_features.tc_norm_detour_time == pytest.approx(0.01691548)
    assert trip_cost_features.tc_norm_idling_time == pytest.approx(0.6011666)
    assert trip_cost_features.tc_sharing_efficiency_factor == pytest.approx(0.614379)

@pytest.mark.basic
def test_trip_cost_features_double(config: Config, payload: dict, trip_costs_neworder: list[TripCost]):
    fb = FeatureBuilder(payload, config)

    current_time = 20000
    trip_cost = trip_costs_neworder[7]

    feat_dict = fb._trip_cost_features(trip_cost, current_time)
    trip_cost_features = TripCostFeatures(**feat_dict)

    # TripCost(trip_no=8, vehicle_id=0, cost=1063.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], plan=TripInsertionPlan(depot_feasible=True, sequence_feasible=True, added_cost=1063.1, sequence=[VehicleStop(trip_id='2-5', node=Node(lat=35.720035553, lon=-77.893722534), type='pickup', dwell=180), VehicleStop(trip_id='2-2', node=Node(lat=35.75359726, lon=-77.924423218), type='pickup', dwell=180), VehicleStop(trip_id='2-5', node=Node(lat=35.753082275, lon=-77.930335999), type='dropoff', dwell=60), VehicleStop(trip_id='2-2', node=Node(lat=35.740951538, lon=-77.963066101), type='dropoff', dwell=60)], trips=[Trip(request_id=2, trip_number=1, am_capacity=1, wc_capacity=0, pick_up_time=19827, latest_pick_up_time=21627, earliest_arrival_time=20270.9, latest_arrival_time=22070.9, origin=Node(lat=35.75359726, lon=-77.924423218), destination=Node(lat=35.740951538, lon=-77.963066101), dwell_pickup=180, dwell_alight=60, iteration=2, ), Trip(request_id=5, trip_number=4, am_capacity=1, wc_capacity=0, pick_up_time=20002, latest_pick_up_time=21802, earliest_arrival_time=20451.0, latest_arrival_time=22251.0, origin=Node(lat=35.720035553, lon=-77.893722534), destination=Node(lat=35.753082275, lon=-77.930335999), dwell_pickup=180, dwell_alight=60, iteration=2, )], next_immediate_node=Node(lat=35.723017652422435, lon=-77.90871990823223), time_at_next_immediate_node=18922, veh_travel_time=148.1, depot_travel_time=463.2, direct_trip_times=[445.9, 449.0], total_direct_travel_time=894.9, actual_travel_time=1063.1, total_dwell_time=480.0, actual_route_travel_time=1543.1, detour_time=168.19999999999993, idling_time=756.9)), 

    assert trip_cost_features.tc_cost == pytest.approx(1063.1)
    assert trip_cost_features.tc_sequence_len == 4
    assert trip_cost_features.tc_num_trips == 2
    assert trip_cost_features.tc_total_am_demand == 2
    assert trip_cost_features.tc_total_wc_demand == 0

    assert trip_cost_features.tc_norm_travel_time_to_first_pickup == pytest.approx(0.014026778870117414)
    assert trip_cost_features.tc_total_direct_travel_time == pytest.approx(894.9)
    assert trip_cost_features.tc_actual_travel_time == pytest.approx(1063.1)
    assert trip_cost_features.tc_norm_batch_actual_travel_time == pytest.approx(0.88591667)
    assert trip_cost_features.tc_total_dwell_time == pytest.approx(480.0)
    assert trip_cost_features.tc_dwell_time_ratio == pytest.approx(0.45150974)
    assert trip_cost_features.tc_detour_time == pytest.approx(168.19999999999993)
    assert trip_cost_features.tc_norm_detour_time == pytest.approx(0.015930480796446645)
    assert trip_cost_features.tc_norm_idling_time == pytest.approx(0.63075)
    assert trip_cost_features.tc_sharing_efficiency_factor == pytest.approx(0.18795396)