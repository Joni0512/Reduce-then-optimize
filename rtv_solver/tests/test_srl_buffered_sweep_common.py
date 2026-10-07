"""Tests for the buffered-actor wandb sweep helpers (2026-10-07), rtv_solver/pipeline/srl_buffered_sweep_common.py."""
import json

import pytest

from rtv_solver.pipeline.srl_buffered_sweep_common import log_outcome, parse_slurm_elapsed, read_trial_outcome


class FakeWandb:
    def __init__(self):
        self.logged = []

    def log(self, data):
        self.logged.append(data)


def _curve():
    return [{"epoch": 5, "service_rate": 0.5}, {"epoch": 10, "service_rate": 0.6}]


@pytest.mark.basic
def test_outcome_done_prefers_result_json(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"best_epoch": 10, "best_val_service_rate": 0.6, "test_service_rate": 0.7, "val_curve": _curve()}))
    (tmp_path / "partial_val_curve.json").write_text(json.dumps({"best_epoch": 5}))
    status, payload = read_trial_outcome(tmp_path)
    assert status == "done" and payload["test_service_rate"] == 0.7


@pytest.mark.basic
def test_outcome_partial_and_failed(tmp_path):
    assert read_trial_outcome(tmp_path) == ("failed", None)
    (tmp_path / "partial_val_curve.json").write_text(json.dumps({"best_epoch": 5, "best_val_service_rate": 0.5, "last_epoch": 10, "val_curve": _curve()}))
    status, payload = read_trial_outcome(tmp_path)
    assert status == "partial" and payload["last_epoch"] == 10


@pytest.mark.basic
def test_log_outcome_never_logs_fake_zero_metric_for_failures():
    fake = FakeWandb()
    log_outcome(fake, "failed", None, extra={"run_id": "x"})
    assert fake.logged == [{"trial_failed": True, "run_id": "x"}]  # no best_val_service_rate=0.0

    fake = FakeWandb()
    log_outcome(fake, "partial", {"best_epoch": 5, "best_val_service_rate": 0.5, "last_epoch": 10, "val_curve": _curve()})
    assert fake.logged[0]["trial_failed"] is True and "test_service_rate" not in fake.logged[0]
    assert [d["epoch"] for d in fake.logged[1:]] == [5, 10]

    fake = FakeWandb()
    log_outcome(fake, "done", {"best_epoch": 10, "best_val_service_rate": 0.6, "test_service_rate": 0.7, "val_curve": _curve()})
    assert fake.logged[0]["trial_failed"] is False and fake.logged[0]["test_service_rate"] == 0.7


@pytest.mark.basic
@pytest.mark.parametrize("text,seconds", [("00:01:13", 73), ("20:48:03", 74883), ("1-02:00:00", 93600)])
def test_parse_slurm_elapsed(text, seconds):
    assert parse_slurm_elapsed(text) == seconds
