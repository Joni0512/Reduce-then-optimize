from types import SimpleNamespace
from rtv_solver.pipeline.mc_return_builder import MonteCarloReturnBuilder


def test_local_served_windows():
    b = MonteCarloReturnBuilder([])
    # windows start at 0,100,200; pickups at 50 (->0), 100 (->1, boundary belongs to the later window), 250 (->2, open-ended last), -5 (->0, before t_0)
    r = b.build_local_served([0, 100, 200], {1: 50, 2: 100, 3: 250, 4: -5})
    assert r == [2.0, 1.0, 1.0]
    assert sum(r) == 4


def test_local_served_empty():
    assert MonteCarloReturnBuilder([]).build_local_served([], {1: 5}) == []
