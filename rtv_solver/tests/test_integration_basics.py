"""
`pytest -q -m basic`

The following integration tests have been manually tested and returned valid results. These specific results of the statistics outcome are tested and confirmed here.
"""

import pickle
import pytest
from pathlib import Path

from rtv_solver import OfflineRTVSolver, OnlineRTVSolver, COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.structure.config import Config

def _init_payload(vehicle_count: int = 1, first_vehicle_reduced_time: int = 72000, request_time_span_minutes: int = 20) -> dict:
    """
    Payload initializer with some basic restrictions that one can change accordingly

    :param int vehicle_count: Reduction of drivers to first ( 0 <> vehicle_count ) vehicles
    :param int first_vehicle_reduced_time: Reduce end_time of vehicle 0
    :param int time_span_minutes: Reduce or increase time of which requests are considered 
    """
    TEST_DIR = Path(__file__).resolve().parent.parent
    INPUTS_DIR = TEST_DIR.parent / "inputs"
    path = INPUTS_DIR / "wilson_nc_initial.pkl"
    # input_file = "rtv-solver/inputs/wilson_nc_initial.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = PayloadParser.normalize_to_canonical(data)

    driver_runs_total = data[PayloadParser.DRIVERS]
    driver_runs_reduced = driver_runs_total[:vehicle_count]
    vehicle_state = driver_runs_reduced[0][PayloadParser.DRIVER_STATE]
    vehicle_manifest = driver_runs_reduced[0][PayloadParser.DRIVER_MANIFEST]
    vehicle_state[PayloadParser.DRIVER_STATE_END_TIME] = first_vehicle_reduced_time 

    current_time = 5 * 3600 + 30 * 60
    step = request_time_span_minutes * 60

    selected_requests = []
    for request in data[PayloadParser.REQUESTS]:
        if request[PayloadParser.REQ_PICKUP_WINDOW_START] < current_time + step:
            selected_requests.append(request)

    payload = {
        PayloadParser.DEPOT: data[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: selected_requests,
        PayloadParser.DRIVERS: driver_runs_reduced,
    }
    return payload

@pytest.mark.basic
@pytest.mark.integration
@pytest.mark.offline    
def test_integration_offlineRTVsolver_vehicle1_maxCard3():
    """
    Integration test for a known run with 1 vehicle and max_cardinality = 3, reduced time
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.MAX_CARDINALITY = 3
    config.STEP_SIZE = 600
    config.BATCH_INTERVAL = 600
    config.MODE = 'offline'
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs = off_solver.solve_rtv(
        payload,
        config.BATCH_INTERVAL,
        config.STEP_SIZE,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload)

    # Test assertions
    assert feasible is True
    assert violations == []

    assert stats.vmt == pytest.approx(1070.5)
    assert stats.pmt == pytest.approx(1501.1)
    assert stats.serviced == 4

    assert stats.wait_time == pytest.approx([0, 705.7999999999993, 400.2999999999993, 1123.7999999999993])
    assert stats.detour == pytest.approx([800.2, 664.7999999999993, 771.5, 299.99999999999926])
    assert stats.vmt_over_pmt == pytest.approx(0.7131436946239424)
    assert stats.average_wait_time == pytest.approx(557.4749999999995)
    assert stats.average_detour == pytest.approx(634.1249999999995)

@pytest.mark.basic
@pytest.mark.integration
@pytest.mark.offline
def test_integration_offlineRTVsolver_vehicle2_maxCard2():
    """
    Integration test for a known run with 2 vehicles and max_cardinality = 2
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=2, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.MAX_CARDINALITY = 2
    config.STEP_SIZE = 600
    config.BATCH_INTERVAL = 600
    config.MODE = 'offline'
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs = off_solver.solve_rtv(
        payload,
        config.BATCH_INTERVAL,
        config.STEP_SIZE,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload)

    # Test assertions
    assert feasible is True
    assert violations == []

    assert stats.vmt == pytest.approx(2439.6999999999994)
    assert stats.pmt == pytest.approx(2656.2999999999997)
    assert stats.serviced == 8

    # sometimes wait_time has different order, FIX sorted lists (avg is more important test)
    assert sorted(stats.wait_time) == pytest.approx(sorted([0, 204.5, 70.90000000000146, 955.3000000000029, 178.59999999999854, 0, 0, 730.6999999999971]))
    assert sorted(stats.detour) == pytest.approx(sorted([1292.0000000000036, 1140.7000000000037, 180.0, 370.0, 774.3, 495.79999999999853, 693.0999999999979, 318.0999999999993]))

    assert stats.vmt_over_pmt == pytest.approx(0.918458005496367)
    assert stats.average_wait_time == pytest.approx(267.5)
    assert stats.average_detour == pytest.approx(658.0000000000003)

@pytest.mark.basic
@pytest.mark.integration
@pytest.mark.online
def test_integration_onlineRTVsolver_vehicle3_maxCard3():
    """
    Integration test for an online RTV solver run with 3 vehicles and max_cardinality = 3
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=3, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.MAX_CARDINALITY = 3
    config.STEP_SIZE = 600
    config.BATCH_INTERVAL = 600
    config.MODE = 'online'
    # run solver
    on_solver = OnlineRTVSolver(config)
    updated_driver_runs, _ = on_solver.solve_pdptw_rtv(payload)
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload)

    # Test assertions
    assert feasible is True
    assert violations == []

    assert stats.vmt == pytest.approx(2595.7999999999997)
    assert stats.pmt == pytest.approx(3085.5)
    assert stats.serviced == 9

    # sometimes wait_time has different order, FIX sorted lists (avg is more important test)
    assert sorted(stats.wait_time) == pytest.approx(sorted([0, 822.5999999999985, 1027.0999999999985, 0, 0, 0, 0, 0, 208.20000000000073]))
    assert sorted(stats.detour) == pytest.approx(sorted([1015.0999999999985, 391.2999999999993, 239.99999999999926, 952.8999999999985, 495.79999999999853, 388.49999999999926, 1495.8000000000006, 639.0000000000007, 179.99999999999926]))

    assert stats.vmt_over_pmt == pytest.approx(0.8412899043915086)
    assert stats.average_wait_time == pytest.approx(228.6555555555553)
    assert stats.average_detour == pytest.approx(644.266666666666)

@pytest.mark.basic
@pytest.mark.integration
@pytest.mark.rh
def test_integration_RHsolver_vehicleDeactivation_keepActiveTrue():
    """
    Integration edge case with specific vehicle that ends before requests are finished

    ensures that assignment handles vehicles and active requests correctly that are close to being inactive  

    FIXME performance increase if we removed any vehicles from the TripGeneration once they are inactive and do not iterate if our vehicle are not active anymore
    """
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=22000, request_time_span_minutes=90)
    config = Config()
    config.MAX_CARDINALITY = 3
    config.STEP_SIZE = 1200
    config.BATCH_INTERVAL = 3600
    config.KEEP_ACTIVE = True
    config.RETURN_DEPOT = True
    config.MODE = 'offline'
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs = off_solver.solve_rtv(
        payload,
        config.BATCH_INTERVAL,
        config.STEP_SIZE,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload)

    # Test assertions
    assert feasible is True
    assert violations == []

    assert stats.vmt == pytest.approx(1051.1999999999998)
    assert stats.pmt == pytest.approx(897.8)
    assert stats.serviced == 3

    # sometimes wait_time has different order, FIX sorted lists (avg is more important test)
    assert sorted(stats.wait_time) == pytest.approx(sorted([0, 0, 0]))
    assert sorted(stats.detour) == pytest.approx(sorted([952.8999999999985, 495.79999999999853, 388.49999999999926]))

    assert stats.vmt_over_pmt == pytest.approx(1.1708621073735797)
    assert stats.vmt_over_pmt_woDepot == pytest.approx(0.9680329694809533)
    assert stats.average_wait_time == pytest.approx(0.0)
    assert stats.average_detour == pytest.approx(612.3999999999988)

    assert stats.depot_movements == 1
    assert stats.total_requests == 47

@pytest.mark.basic
@pytest.mark.integration
@pytest.mark.coaml
def test_integration_COAMLsolver_vehicle1_maxCard3():
    """
    Integration test for a known run with 1 vehicle and max_cardinality = 3
    """
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=72000, request_time_span_minutes=5)
    config = Config()
    config.MAX_CARDINALITY = 3
    config.STEP_SIZE = 600
    config.BATCH_INTERVAL = 600
    config.MODE = 'rh-ml'

    rh_solver = COAMLPipeline(config, payload)
    updated_driver_runs = rh_solver.solve_pdptw(payload)

    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload)

    # Simplest test to check whether it runs at all correctly
    assert feasible is True
    assert violations == []

# if __name__ == '__main__':
    # if we require a test of the results here, just add it to the main loop and start a debug run

    # DEBUGGING