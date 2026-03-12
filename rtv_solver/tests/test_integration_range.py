import pytest

from rtv_solver.offline_rtv_solver import OfflineRTVSolver

from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.handlers.stats_parser import StatsParser

from rtv_solver.structure.config import Config

from rtv_solver.tests.test_integration_basics import _init_payload

"""
The following integration tests run rolling horizon approaches across different configurations. We hope to find issues with certain time ranges or combinations of configurations.

Add configurations that once returned errors so we can update the tool accordingly.
"""


@pytest.mark.parametrize(
    "vehicle_count," \
    "first_vehicle_reduced_time," \
    "request_time_span_minutes," \
    "max_cardinality," \
    "batch_interval," \
    "step_size," \
    "keep_active," \
    "return_depot," \
    "expected_feasible" ,
    [   
        pytest.param(
            1, 22_000, 90, 3, 3600, 1200, True, True,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-1"
        ),
        pytest.param(
            1, 22_000, 90, 3, 3600, 1200, False, True,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-2"
        ),
        pytest.param(
            1, 22_000, 90, 3, 3600, 1200, False, False,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-3"
        ),
        pytest.param(
            1, 22_000, 90, 3, 3600, 1200, True, False,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-3"
        ),
        pytest.param(
            1, 72000, 20, 2, 1200, 600, True, False,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-5"
        ),
        pytest.param(
            2, 72_000, 60, 3, 1200, 600, True, False,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-6"
        ),
        pytest.param(
            1, 21000, 60, 3, 3600, 1200, True, True,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-7"
        ),
        pytest.param(
            2, 21000, 60, 3, 3600, 1200, True, True,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-8"
        ),
        pytest.param(
            2, 30000, 60, 3, 3600, 1200, True, True,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-9"
        ),
        pytest.param(
            1, 25000, 90, 3, 3600, 1200, True, True,      True,
            marks=[pytest.mark.integration, pytest.mark.rh],
            id = "rh-10"
        ),
    ],)
@pytest.mark.server
def test_integration_solver_parametrized(
        vehicle_count: int, 
        first_vehicle_reduced_time: int, 
        request_time_span_minutes: int,
        max_cardinality: int,
        batch_interval: int,
        step_size: int,
        keep_active: bool,
        return_depot: bool,
        expected_feasible: bool,
        config: Config):
    """
    Integration edge case with specific vehicle that ends before requests are finished

    ensures that assignment handles vehicles and active requests correctly that are close to being inactive

    FIXME performance increase if we removed any vehicles from the TripGeneration once they are inactive and do not iterate if our vehicle are not active anymore
    """
    payload = _init_payload(vehicle_count, first_vehicle_reduced_time, request_time_span_minutes)
    # update parameters of config for constantly fixed results
    config.MAX_CARDINALITY = max_cardinality
    config.STEP_SIZE = step_size
    config.BATCH_INTERVAL = batch_interval
    config.KEEP_ACTIVE = keep_active
    config.RETURN_DEPOT = return_depot
    config.SERVER_URL = "http://127.0.0.1:5001/" # tests work only with active server

    expected_depot_movements = 0
    if return_depot:
        expected_depot_movements = vehicle_count
    
    # run solver
    off_solver = OfflineRTVSolver(config)
    updated_driver_runs = off_solver.solve_rtv(
        payload,
        config.BATCH_INTERVAL,
        config.STEP_SIZE,
    )
    # compute stats
    stats_payload = {
        PayloadKeys.DEPOT: payload[PayloadKeys.DEPOT],
        PayloadKeys.REQUESTS: payload[PayloadKeys.REQUESTS],
        PayloadKeys.DRIVERS: updated_driver_runs,
        PayloadKeys.TIME_MATRIX: payload.get(PayloadKeys.TIME_MATRIX, None)}
    stats_evaluator = StatsParser(config=config, payload=stats_payload)
    feasible, stats, violations = stats_evaluator.evaluate(stats_payload)

    # Test assertions
    assert feasible is expected_feasible
    assert stats.depot_movements == expected_depot_movements