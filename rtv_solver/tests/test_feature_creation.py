import pytest
from rtv_solver.pipeline import FeatureBuilder
from rtv_solver.pipeline.feat_builder import VehicleFeatures
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.structure.config import Config

@pytest.mark.basic
def test_vehicle_features(payload: dict, vehicles_interim: dict[int, Vehicle]):
    config = Config()
    fb = FeatureBuilder(payload, config)

    current_time = 20000
    vehicle = vehicles_interim[1]

    feat_dict = fb._vehicle_features(vehicle, vehicles_interim, current_time)
    
    vehicle_features = VehicleFeatures(**feat_dict)
    # simplified to turn dict into a vehicle_features-object to improve assert
    assert vehicle_features.norm_lat_next_position == pytest.approx(0.23299847)
    assert vehicle_features.norm_lon_next_position == pytest.approx(0.193618035)
    assert vehicle_features.norm_interval_remaining_boarded_time == (22832.1 - current_time) / config.BATCH_INTERVAL
    assert vehicle_features.norm_remaining_operating_period == pytest.approx(0.658067053)
    assert vehicle_features.norm_step_remaining_boarded_time == (22832.1 - current_time) / config.STEP_SIZE
    assert vehicle_features.norm_remaining_am_cap == 0.875
    assert vehicle_features.norm_remaining_wc_cap == 1.0
    assert vehicle_features.norm_vehicle_count_in_proximity == 0.0
    assert vehicle_features.avg_vehicle_distance == pytest.approx(0.7553968)
