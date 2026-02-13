import pytest

from rtv_solver.structure.vehicle import Vehicle


def test_vehicle0_capacity_calculation(vehicle_0: Vehicle):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_0.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 1.0
    assert remain_wc_cap == 1.0

def test_vehicle1_capacity_calculation(vehicle_1: Vehicle):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_1.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 0.875
    assert remain_wc_cap == 1.0

def test_vehicle_1_capacities(vehicle_1: Vehicle):
    am_used, wc_used, am_capacity, wc_capacity = vehicle_1.get_capacities()

    assert am_used == 1
    assert wc_used == 0
    assert am_capacity == 8
    assert wc_capacity == 3
    
def test_vehicle2_capacity_calculation(vehicle_2: Vehicle):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_2.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 0.75
    assert remain_wc_cap == 1.0

def test_vehicle_2_capacities(vehicle_2: Vehicle):
    am_used, wc_used, am_capacity, wc_capacity = vehicle_2.get_capacities()

    assert am_used == 2
    assert wc_used == 0
    assert am_capacity == 8
    assert wc_capacity == 3

def test_vehicle3_capacity_calculation(vehicle_3):
    am_cap, wc_cap, remain_am_cap, remain_wc_cap = vehicle_3.get_remaining_capacities()

    assert am_cap == 8
    assert wc_cap == 3
    assert remain_am_cap == 0.625
    assert remain_wc_cap == 1.0

def test_vehicle_3_capacities(vehicle_3: Vehicle):
    am_used, wc_used, am_capacity, wc_capacity = vehicle_3.get_capacities()

    assert am_used == 3
    assert wc_used == 0
    assert am_capacity == 8
    assert wc_capacity == 3