import pickle
import pytest

from rtv_solver import OfflineRTVSolver, OnlineRTVSolver
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.structure.config import Config

# tests are built to run quickly and meant to test the general variants of the code base (extrapolating combinations such as different cardinality or solver approach (online, offline, rolling horizon)
# This code is not meant for performance testing.

def _init_payload(vehicle_count: int = 1, first_vehicle_reduced_time: int = 72000, request_time_span_minutes: int = 20) -> dict:
    """
    Payload initializer with some basic restrictions that one can change accordingly

    :param int vehicle_count: Reduction of drivers to first ( 0 <> vehicle_count ) vehicles
    :param int first_vehicle_reduced_time: Reduce end_time of vehicle 0
    :param int time_span_minutes: Reduce or increase time of which requests are considered 
    """
    input_file = "rtv-solver/inputs/wilson_nc_initial.pkl"
    with open(input_file, "rb") as f:
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
    
def test_integration_offlineRTVsolver_vehicle1_maxCard3():
    """
    Integration test for a known run with 1 vehicle and max_cardinality = 3, reduced time
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.max_cardinality = 3
    config.step_size = 600
    config.batch_interval = 600
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

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

def test_integration_offlineRTVsolver_vehicle2_maxCard2():
    """
    Integration test for a known run with 2 vehicles and max_cardinality = 2
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=2, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.max_cardinality = 2
    config.step_size = 600
    config.batch_interval = 600
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

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

def test_integration_onlineRTVsolver_vehicle3_maxCard3():
    """
    Integration test for an online RTV solver run with 3 vehicles and max_cardinality = 3
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=3, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.max_cardinality = 3
    config.step_size = 600
    config.batch_interval = 600
    # run solver
    on_solver = OnlineRTVSolver(config)
    updated_driver_runs, requests_development = on_solver.solve_pdptw_rtv(payload)
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, requests_development)

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

def test_integration_RHsolver_vehicle1_maxCard2_interval1200():
    """
    Integration test for a known run with 1 vehicle, max_cardinality = 2 and batch_interval = 1200
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.max_cardinality = 2
    config.step_size = 600
    config.batch_interval = 1200
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

    # Test assertions
    assert feasible is True    
    # TODO add actual results when it is run through

def test_integration_RHsolver_vehicle3_maxCard2_interval1200():
    """
    Integration test for a known run with 2 vehicle, max_cardinality = 2 and batch_interval = 1200
    """
    # initialize data and config
    payload = _init_payload(vehicle_count=2, first_vehicle_reduced_time=72000, request_time_span_minutes=20)
    config = Config()
    config.max_cardinality = 2
    config.step_size = 600
    config.batch_interval = 1200
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

    # Test assertions
    assert feasible is True    
    # TODO add actual results when it is run through

def test_integration_RHsolver_depotReturn():
    """
    Config edge case where the vehicles have to return to the depot afterwards

    FIXME iterations keep running and still try to optimize despite no active vehicle being left
    """
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=21000, request_time_span_minutes=60)
    
    config = Config()
    config.max_cardinality = 2
    config.step_size = 1200
    config.batch_interval = 3600
    config.return_depot = True
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

    # Test assertions
    assert feasible is True 
    assert stats.depot_movements == 1   
    # TODO add actual results when it is run through

def test_integration_RHsolver_depotReturn_2vehicles():
    """
    Config edge case where the vehicles have to return to the depot afterwards; use two vehicles but different timeframes
    Integration edge case with specific vehicle and requests, main result is whether vehicles return to depot if they should
    
    TODO how to set vehicles to inactive, so they are not part of the optimization anymore but are also completed in their manifest (depot return and complete manifest of prior assigned trips)
    """
    payload = _init_payload(vehicle_count=2, first_vehicle_reduced_time=21000, request_time_span_minutes=60)
    
    config = Config()
    config.max_cardinality = 2
    config.step_size = 1200
    config.batch_interval = 3600
    config.return_depot = True
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

    # Test assertions
    assert feasible is True 
    assert stats.depot_movements == 2
    # TODO add actual results when it is run through

def test_integration_RH_solver_depotReturn_shortenedRequests():
    """
    Config edge case where the vehicles have to return to the depot afterward, but we reduce request_time_span to force the vehicle to reject requests to get to the depot
    """
    payload = _init_payload(vehicle_count=2, first_vehicle_reduced_time = 21000, request_time_span_minutes = 20)
    
    config = Config()
    config.max_cardinality = 3
    config.step_size = 1200
    config.batch_interval = 3600
    config.return_depot = True
    config.keep_active = True
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

    # Test assertions
    assert feasible is True 
    assert stats.depot_movements == 2
    # TODO add actual results when it is run through

def test_config_RHsolver_vehicleDeactivation_keepActiveTrue():
    """
    Integration edge case with specific vehicle that ends before requests are finished

    ensures that assignment handles vehicles and active requests correctly that are close to being inactive  
    
    BUG this fails because with keep_active = True, the assignment of active requests happens to already inactive vehicles (not considered in the simulation aynmore?); possibly at 21000 or step_size 1800 the timing just fits that the request is not accepted while we keep a valid solution 

    FIXME performance increase if we removed any vehicles from the TripGeneration once they are inactive and do not iterate if our vehicle are not active anymore
    """
    payload = _init_payload(vehicle_count=1, first_vehicle_reduced_time=22000, request_time_span_minutes=90)
    config = Config()
    config.max_cardinality = 3
    config.step_size = 1200
    config.batch_interval = 3600
    config.keep_active = True
    config.return_depot = True
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs, unserved_requests = off_solver.solve_rtv(
        payload,
        config.batch_interval,
        config.step_size,
    )
    # compute stats
    stats_payload = {
        PayloadParser.DEPOT: payload[PayloadParser.DEPOT],
        PayloadParser.REQUESTS: payload[PayloadParser.REQUESTS],
        PayloadParser.DRIVERS: updated_driver_runs,}
    stats_evaluator = StatsParser(config=config)
    feasible, stats, violations, unserved = stats_evaluator.evaluate(stats_payload, unserved_requests)

    # Test assertions
    assert feasible is True    
    # TODO add actual results when it is run through