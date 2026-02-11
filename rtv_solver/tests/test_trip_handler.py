import pytest

from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.structure.config import Config


@pytest.mark.basic
def test_trip_handler_init_with_empty_vehicles_leaves_state_empty():
    """
    TripHandler(vehicles={}, ...) leaves trips and assignment dicts empty and does not raise.
    After run_generation(), state remains empty (no generation runs when vehicles is empty).
    """
    config = Config()
    trip_handler = TripHandler(
        vehicles={},
        requests=[],
        active_requests={},
        iteration=0,
        config=config,
    )
    assert trip_handler.trips == []
    assert trip_handler.ondemand_only_trip_map == {}
    assert trip_handler.shared_trips_map == {}
    assert trip_handler.vehicle_to_trips_cost_map == {}
    assert trip_handler.trip_to_vehicle_cost_map == {}

    trip_handler.run()
    assert trip_handler.trips == []
    assert trip_handler.ondemand_only_trip_map == {}
