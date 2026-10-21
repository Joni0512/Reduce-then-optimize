import pytest

from rtv_solver.pipeline.srl_training_loop import lr_factor


def test_none_is_constant():
    assert all(lr_factor("none", e, 20, 0.1) == 1.0 for e in range(20))


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_endpoints(schedule):
    assert lr_factor(schedule, 0, 20, 0.1) == pytest.approx(1.0)
    assert lr_factor(schedule, 19, 20, 0.1) == pytest.approx(0.1)


def test_cosine_midpoint_and_monotone():
    vals = [lr_factor("cosine", e, 21, 0.1) for e in range(21)]
    assert vals[10] == pytest.approx(0.55)  # halfway between 1.0 and 0.1
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_unknown_schedule():
    with pytest.raises(ValueError):
        lr_factor("step", 0, 20, 0.1)
