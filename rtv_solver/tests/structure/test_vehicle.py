import pytest

from rtv_solver.structure.vehicle import Vehicle

@pytest.mark.basic
def test_vehicle0_capacity_calculation(vehicle_0: Vehicle):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_0.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 1.0
    assert remain_wc_cap == 1.0

@pytest.mark.basic
def test_vehicle1_capacity_calculation(vehicle_1: Vehicle):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_1.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 0.875
    assert remain_wc_cap == 1.0

@pytest.mark.basic
def test_vehicle_1_capacities(vehicle_1: Vehicle):
    am_used, wc_used, am_capacity, wc_capacity = vehicle_1.get_capacities()

    assert am_used == 1
    assert wc_used == 0
    assert am_capacity == 8
    assert wc_capacity == 3

@pytest.mark.basic 
def test_vehicle2_capacity_calculation(vehicle_2: Vehicle):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_2.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 0.75
    assert remain_wc_cap == 1.0

@pytest.mark.basic
def test_vehicle_2_capacities(vehicle_2: Vehicle):
    am_used, wc_used, am_capacity, wc_capacity = vehicle_2.get_capacities()

    assert am_used == 2
    assert wc_used == 0
    assert am_capacity == 8
    assert wc_capacity == 3

@pytest.mark.basic
def test_vehicle3_capacity_calculation(vehicle_3):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_3.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 0.625
    assert remain_wc_cap == 1.0

@pytest.mark.basic
def test_vehicle_3_capacities(vehicle_3: Vehicle):
    am_used, wc_used, am_capacity, wc_capacity = vehicle_3.get_capacities()

    assert am_used == 3
    assert wc_used == 0
    assert am_capacity == 8
    assert wc_capacity == 3

@pytest.mark.basic
def test_vehicle_0_boarding_time(vehicle_0: Vehicle):
    current_time = 18000
    last_boarded_dropoff_time = vehicle_0.get_remaining_boarded_time(current_time)

    assert last_boarded_dropoff_time == 0.0

@pytest.mark.basic
def test_vehicle_2_boarding_time(vehicle_2: Vehicle):
    current_time = 18000
    last_boarded_dropoff_time = vehicle_2.get_remaining_boarded_time(current_time)

    assert last_boarded_dropoff_time == pytest.approx(21996.399999999998)

# 2026-11-17: wc_capacity == 0 (Li&Lim, SF) - the old guard zeroed all four values; per_type=True fixes it,
# per_type=False (default) must keep the old behaviour for checkpoint reproducibility.
@pytest.mark.basic
def test_vehicle1_remaining_capacities_without_wc_seats(vehicle_1: Vehicle):
    vehicle_1.wc_capacity = 0
    assert vehicle_1.get_remaining_capacities() == (0, 0, 0, 0)
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_1.get_remaining_capacities(per_type=True)
    assert am_cap == 8
    assert wc_cap == 0
    assert remain_am_cap == 0.875
    assert remain_wc_cap == 0.0


@pytest.mark.basic
def test_vehicle1_remaining_capacities_per_type_matches_old_when_both_types_exist(vehicle_1: Vehicle):
    assert vehicle_1.get_remaining_capacities(per_type=True) == vehicle_1.get_remaining_capacities()
