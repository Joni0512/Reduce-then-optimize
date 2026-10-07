"""
2026-10-07: shared helpers of the buffered-actor wandb sweep (local agent + cluster coordinator), see
sweep_srl_buffered.yaml. Reads what a finished/crashed trial left in its output dir and logs it to wandb:
- result.json            -> trial finished normally ("done")
- partial_val_curve.json -> trial crashed/timed out after >= 1 validation ("partial", written by
                            run_srl_training_loop after every validation)
- neither                -> "failed"
Failed/partial trials are logged with trial_failed=True and WITHOUT a fake 0.0 metric (the old coordinators
logged 0.0, which polluted the sweep statistics).
"""
import json
from pathlib import Path


def read_trial_outcome(output_dir: Path) -> tuple[str, dict | None]:
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            return "done", json.load(f)
    partial_path = output_dir / "partial_val_curve.json"
    if partial_path.exists():
        with open(partial_path) as f:
            return "partial", json.load(f)
    return "failed", None


def log_outcome(wandb, status: str, payload: dict | None, extra: dict | None = None) -> None:
    """Log one trial's outcome to the active wandb run (caller does wandb.finish)."""
    extra = extra or {}
    if status == "done":
        wandb.log({
            "best_epoch": payload["best_epoch"],
            "best_val_service_rate": payload["best_val_service_rate"],
            "test_service_rate": payload["test_service_rate"],
            "trial_failed": False, **extra,
        })
    elif status == "partial":
        # best val seen before the crash; no test rate (computed only at the end of a full run)
        wandb.log({
            "best_epoch": payload["best_epoch"],
            "best_val_service_rate": payload["best_val_service_rate"],
            "trial_failed": True, "last_epoch": payload["last_epoch"], **extra,
        })
    else:
        wandb.log({"trial_failed": True, **extra})
    if payload is not None:
        for point in payload["val_curve"]:
            wandb.log({"val_service_rate_curve": point["service_rate"], "epoch": point["epoch"]})


def parse_slurm_elapsed(text: str) -> float:
    """sacct Elapsed ('[D-]HH:MM:SS') -> seconds."""
    text = text.strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    h, m, s = (int(x) for x in text.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s
