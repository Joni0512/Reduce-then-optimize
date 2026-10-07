"""
2026-10-07: login-node coordinator that feeds the buffered-actor wandb sweep (sweep_srl_buffered.yaml) with
CLUSTER trials; the local Mac agent (srl_buffered_sweep_local_trial.py) pulls from the same sweep. Compute nodes
have no internet, so this lightweight process (login node, via nohup) talks to wandb, submits one sbatch job per
trial, waits for it, reads the trial's output dir and logs the outcome (same pattern as
srl_twin_critic_no_warmstart_sweep_coordinator.py).

Differences to the old coordinator: failed/partial trials are logged WITHOUT a fake 0.0 metric (see
srl_buffered_sweep_common.py); a trial that dies EARLY (< RETRY_MAX_ELAPSED_S, e.g. a token-server outage during
pretraining) is resubmitted once - a timeout after ~24 h is NOT resubmitted (it would just time out again).

Usage (login node):
    ./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_sweep_coordinator <sweep_id> <count> <cm4|serial>
Run several of these in parallel (one per desired concurrent cluster job; cm4_tiny allows 4 jobs per user).
"""
import subprocess
import sys
import time
import uuid

import wandb

from rtv_solver.pipeline.srl_buffered_sweep_common import read_trial_outcome, log_outcome, parse_slurm_elapsed

EPOCHS = 20
POLL_INTERVAL_SECONDS = 120
MAX_ATTEMPTS = 2
RETRY_MAX_ELAPSED_S = 6 * 3600
SBATCH_SCRIPTS = {
    "cm4": "submit_srl_buffered_actor_lr_trial.sbatch",
    "serial": "submit_srl_buffered_actor_lr_trial_serial.sbatch",
}
CLUSTER = "cm4"  # set from argv in __main__


def submit_job(run_id: str, actor_lr: float, critic_lr: float, seed: int) -> str:
    result = subprocess.run(
        ["sbatch", "-M", CLUSTER,
         f"--export=ALL,SRL_RUN_ID={run_id},ACTOR_LR={actor_lr},CRITIC_LR={critic_lr},SEED={seed},EPOCHS={EPOCHS}",
         SBATCH_SCRIPTS[CLUSTER]],
        capture_output=True, text=True, check=True,
    )
    job_id = result.stdout.strip().split()[3]
    print(f"[coordinator] submitted {CLUSTER} job {job_id} for run_id={run_id} actor_lr={actor_lr:.6f} critic_lr={critic_lr:.6f} seed={seed}")
    return job_id


def wait_for_job(job_id: str) -> None:
    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        result = subprocess.run(["squeue", "-M", CLUSTER, "-j", job_id, "-h", "-o", "%T"], capture_output=True, text=True)
        state = result.stdout.strip()
        if not state:
            print(f"[coordinator] job {job_id} no longer in queue - assuming finished")
            return
        print(f"[coordinator] job {job_id} state={state}, still waiting...")


def job_elapsed_seconds(job_id: str) -> float | None:
    result = subprocess.run(["sacct", "-M", CLUSTER, "-j", job_id, "-X", "-n", "-o", "Elapsed"], capture_output=True, text=True)
    text = result.stdout.strip().split("\n")[0].strip() if result.stdout.strip() else ""
    try:
        return parse_slurm_elapsed(text)
    except ValueError:
        return None


def run_cluster_trial() -> None:
    wandb.init()
    config = wandb.config
    from rtv_solver.pipeline.srl_training_loop import REPO_ROOT
    status, payload, attempts = "failed", None, 0
    run_id = f"bufsweep_{CLUSTER}_{uuid.uuid4().hex[:8]}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        attempt_run_id = run_id if attempt == 1 else f"{run_id}_retry{attempt - 1}"
        job_id = submit_job(attempt_run_id, config.actor_lr, config.critic_lr, config.seed)
        wait_for_job(job_id)
        status, payload = read_trial_outcome(REPO_ROOT / "outputs" / "srl_training_sweep" / attempt_run_id)
        if status == "done":
            run_id = attempt_run_id
            break
        elapsed = job_elapsed_seconds(job_id)
        print(f"[coordinator] {attempt_run_id}: status={status}, elapsed={elapsed}s")
        if attempt < MAX_ATTEMPTS and elapsed is not None and elapsed < RETRY_MAX_ELAPSED_S:
            print("[coordinator] early death - resubmitting once")
            continue
        run_id = attempt_run_id
        break
    log_outcome(wandb, status, payload, extra={"run_id": run_id, "where": CLUSTER, "attempts": attempts})
    print(f"[coordinator] {run_id} finished with status={status}")
    wandb.finish(exit_code=0 if status == "done" else 1)


if __name__ == "__main__":
    sweep_id = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    CLUSTER = sys.argv[3] if len(sys.argv) > 3 else "cm4"
    if CLUSTER not in SBATCH_SCRIPTS:
        raise SystemExit(f"cluster must be one of {sorted(SBATCH_SCRIPTS)}, got {CLUSTER!r}")
    wandb.agent(sweep_id, function=run_cluster_trial, count=count)
