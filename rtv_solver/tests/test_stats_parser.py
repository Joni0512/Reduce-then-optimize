import pytest
from pathlib import Path

from rtv_solver.handlers.stats_parser import StatsParser, run_stats_from_manifest_file


@pytest.mark.basic
def test_compute_detour_time_direct_trip_returns_zero():
    detour = StatsParser._compute_detour_time(
        pickup_service_end_time=100.0,
        dropoff_time=160.0,
        direct_travel_time=60.0,
    )
    assert detour == pytest.approx(0.0)


@pytest.mark.basic
def test_compute_detour_time_positive_extra_ride_time():
    detour = StatsParser._compute_detour_time(
        pickup_service_end_time=100.0,
        dropoff_time=190.0,
        direct_travel_time=60.0,
    )
    assert detour == pytest.approx(30.0)


@pytest.mark.basic
def test_compute_detour_time_negative_raises_assertion():
    with pytest.raises(AssertionError, match="Detour time cannot be negative"):
        StatsParser._compute_detour_time(
            pickup_service_end_time=100.0,
            dropoff_time=150.0,
            direct_travel_time=60.0,
        )


@pytest.mark.basic
def test_compute_detour_time_ignores_pre_pickup_wait():
    # Pickup happens later, but detour only uses time after pickup.
    detour = StatsParser._compute_detour_time(
        pickup_service_end_time=300.0,
        dropoff_time=410.0,
        direct_travel_time=90.0,
    )
    assert detour == pytest.approx(20.0)


@pytest.mark.basic
def test_compute_dropoff_goal_lateness_positive():
    lateness = StatsParser._compute_dropoff_tw_lateness(
        dropoff_service_start=210.0,
        dropoff_tw_start=180.0,
    )
    assert lateness == pytest.approx(30.0)


@pytest.mark.basic
def test_compute_dropoff_goal_lateness_on_time_returns_zero():
    lateness = StatsParser._compute_dropoff_tw_lateness(
        dropoff_service_start=180.0,
        dropoff_tw_start=180.0,
    )
    assert lateness == pytest.approx(0.0)


@pytest.mark.basic
def test_compute_dropoff_goal_lateness_negative_raises_assertion():
    with pytest.raises(AssertionError, match="Lateness cannot be negative"):
        lateness =StatsParser._compute_dropoff_tw_lateness(
            dropoff_service_start=10.0,
            dropoff_tw_start=180.0,
        )


@pytest.mark.basic
def test_compute_waiting_before_service_positive():
    waiting = StatsParser._compute_waiting_before_service(service_start=100.0, computed_arrival_time=80.0)
    assert waiting == pytest.approx(20.0)


@pytest.mark.basic
def test_compute_waiting_before_service_zero():
    waiting = StatsParser._compute_waiting_before_service(service_start=100.0, computed_arrival_time=100.0)
    assert waiting == pytest.approx(0.0)


@pytest.mark.basic
def test_compute_waiting_before_service_negative_raises_assertion():
    with pytest.raises(AssertionError, match="Waiting before service cannot be negative"):
        StatsParser._compute_waiting_before_service(service_start=80.0, computed_arrival_time=100.0)


@pytest.mark.basic
def test_compute_service_dwell_positive():
    dwell = StatsParser._compute_service_dwell(service_start=100.0, service_end=190.0)
    assert dwell == pytest.approx(90.0)


@pytest.mark.basic
def test_compute_service_dwell_zero():
    dwell = StatsParser._compute_service_dwell(service_start=100.0, service_end=100.0)
    assert dwell == pytest.approx(0.0)


@pytest.mark.basic
def test_compute_service_dwell_negative_raises_assertion():
    with pytest.raises(AssertionError, match="Service dwell cannot be negative"):
        StatsParser._compute_service_dwell(service_start=100.0, service_end=80.0)


@pytest.mark.basic
def test_lilim_lc101_stats_regression():
    """Regression test: stats for LiLim lc101.json must remain stable."""
    manifest_path = Path(__file__).resolve().parent.parent.parent / "solutions" / "li_lim" / "manifests" / "lc101.json"
    feasible, stats, violations = run_stats_from_manifest_file(manifest_path)

    assert feasible is True
    assert violations == []

    assert stats.vmt == pytest.approx(828.9368669428342) # original solution value from benchmark
    assert stats.pmt == pytest.approx(282.29635101241917)
    assert stats.serviced == 53
    assert stats.vmt_over_pmt == pytest.approx(2.936406595302985)
    assert stats.vmt_over_pmt_woDepot == pytest.approx(1.9702139666449594)
    assert stats.average_wait_time == pytest.approx(25.633216824915543)
    assert stats.average_detour == pytest.approx(140.45717214633228)
    assert stats.average_dropoff_goal_lateness == pytest.approx(31.56767861299161)
    assert stats.total_requests == 53
    assert [int(x) for x in stats.serviced_requests] == [81, 78, 76, 71, 79, 57, 54, 53, 56, 98, 96, 92, 97, 100, 13, 18, 19, 16, 32, 33, 35, 38, 36, 90, 87, 86, 82, 84, 43, 42, 44, 45, 51, 50, 49, 67, 65, 63, 62, 64, 66, 5, 3, 8, 11, 9, 6, 20, 25, 29, 30, 28, 23]
    assert stats.depot_movements == 10
    assert stats.depot_vmt == pytest.approx(272.7526534452581)
    assert stats.rebalancing_movements == 0
    assert stats.rebalancing_vmt == pytest.approx(0.0)
    assert stats.total_time == pytest.approx(0.0)

    # hand-checked for requests 81 and 57
    assert stats.wait_time[0] == pytest.approx(0.43416490252568707)
    assert stats.detour[0] == pytest.approx(273.819660112501)
    assert stats.dropoff_goal_lateness[0] == pytest.approx(35.43416490252571)

    assert stats.wait_time[5] == pytest.approx(0.0)
    assert stats.detour[5] == pytest.approx(0.0)
    assert stats.dropoff_goal_lateness[5] == pytest.approx(32.0)

    # check all values are as expected (rel=1e-2, abs=1 for minor float/rounding differences)
    _tol = dict(rel=1e-2, abs=1.0)
    assert stats.wait_time == pytest.approx([0, 31, 29, 34, 32, 0, 36, 31, 26, 1, 31, 37, 31, 40, 1, 39, 35, 28, 1, 37, 31, 23, 27, 1, 31, 34, 27, 33, 1, 42, 28, 30, 31, 34, 34, 0, 27, 24, 28, 31, 26, 0, 41, 36, 30, 37, 42, 0, 28, 25, 29, 25, 24], **_tol)
    assert stats.detour == pytest.approx([274, 0, 188, 183, 0, 0, 276, 90, 185, 91, 90, 0, 0, 0, 0, 366, 0, 0, 92, 190, 182, 184, 0, 560, 90, 559, 90, 182, 91, 91, 0, 0, 0, 0, 0, 459, 274, 92, 369, 0, 0, 91, 835, 0, 363, 90, 91, 0, 0, 183, 363, 182, 0], **_tol)
    assert stats.dropoff_goal_lateness == pytest.approx([35, 31, 37, 35, 27, 32, 34, 32, 46, 22, 27, 22, 31, 39, 26, 36, 24, 28, 19, 27, 30, 34, 27, 32, 38, 28, 29, 33, 36, 30, 31, 31, 31, 30, 73, 40, 28, 30, 22, 31, 28, 28, 37, 28, 29, 28, 24, 40, 28, 41, 27, 37, 24], **_tol)
    