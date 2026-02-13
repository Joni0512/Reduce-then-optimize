import pytest
from rtv_solver.pipeline import FeatureBuilder
from rtv_solver.pipeline.feat_builder import VehicleFeatures, StateFeatures
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.config import Config

@pytest.mark.basic
def test_vehicle_features_start(config, payload: dict, vehicles_start: dict[int, Vehicle]): 
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
def test_vehicle_features_interim(config, payload: dict, vehicles_interim: dict[int, Vehicle]):
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
def test_state_features_start(config, payload, vehicles_start):
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
def test_state_features_interim(config, payload, vehicles_interim):
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