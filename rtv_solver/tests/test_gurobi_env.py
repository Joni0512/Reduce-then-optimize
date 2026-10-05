"""Tests for rtv_solver/util/gurobi_env.py's retry-on-token-server-outage helper (2026-10-05)."""
import gurobipy as gp
import pytest

from rtv_solver.util import gurobi_env

TOKEN_ERR = "Failed to connect to token server 'license1.lrz.de' (port 41954)"


def _fake_env_factory(errors: list[str | None]):
    """gp.Env replacement whose start() raises errors[i] (None = success) on the i-th instance."""
    created = []

    class FakeEnv:
        def __init__(self, empty=True):
            self.index = len(created)
            self.disposed = False
            created.append(self)

        def setParam(self, *_args):
            pass

        def start(self):
            err = errors[min(self.index, len(errors) - 1)]
            if err is not None:
                raise gp.GurobiError(10009, err)

        def dispose(self):
            self.disposed = True

    return FakeEnv, created


@pytest.mark.basic
def test_retries_token_server_error_then_succeeds(monkeypatch):
    fake, created = _fake_env_factory([TOKEN_ERR, TOKEN_ERR, None])
    sleeps = []
    monkeypatch.setattr(gurobi_env.gp, "Env", fake)
    monkeypatch.setattr(gurobi_env.time, "sleep", sleeps.append)

    env = gurobi_env.start_gurobi_env()

    assert env is created[2]
    assert len(created) == 3
    assert sleeps == [5.0, 10.0]
    assert created[0].disposed and created[1].disposed and not created[2].disposed


@pytest.mark.basic
def test_other_gurobi_errors_are_raised_immediately(monkeypatch):
    fake, created = _fake_env_factory(["User name mismatch (licensed to 'joni', current user is '')"])
    sleeps = []
    monkeypatch.setattr(gurobi_env.gp, "Env", fake)
    monkeypatch.setattr(gurobi_env.time, "sleep", sleeps.append)

    with pytest.raises(gp.GurobiError):
        gurobi_env.start_gurobi_env()

    assert len(created) == 1
    assert sleeps == []


@pytest.mark.basic
def test_gives_up_after_max_wait(monkeypatch):
    fake, created = _fake_env_factory([TOKEN_ERR])
    sleeps = []
    monkeypatch.setattr(gurobi_env.gp, "Env", fake)
    monkeypatch.setattr(gurobi_env.time, "sleep", sleeps.append)

    with pytest.raises(gp.GurobiError):
        gurobi_env.start_gurobi_env(max_wait_s=30.0)

    assert sum(sleeps) == pytest.approx(30.0)
    assert len(created) == len(sleeps) + 1
    assert all(e.disposed for e in created)


@pytest.mark.basic
def test_real_env_smoke():
    try:
        env = gurobi_env.start_gurobi_env(max_wait_s=0.0)
    except gp.GurobiError as exc:
        pytest.skip(f"no usable Gurobi license here: {exc}")
    with env:
        model = gp.Model("smoke", env=env)
        model.addVar()
        model.update()
        assert model.NumVars == 1
