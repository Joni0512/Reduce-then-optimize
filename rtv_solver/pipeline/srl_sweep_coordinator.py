"""
2026-09-13: login-node coordinator for the SIL/SRL(local)/SRL(local_positive)
actor_lr x critic_lr wandb random-search sweep - same pattern as
sweep_td_bootstrap_gat_bufferfix_coordinator.py. Compute nodes have no
internet, so THIS process (running on a login node, which has internet)
talks to wandb; the actual training happens in a normal sbatch job
(srl_cluster_trial.py) that never touches wandb.

reward_mode is one of the sweep's own (fixed-value) parameters - see
sweep_srl_local.yaml / sweep_srl_local_positive.yaml - so wandb.config
already carries it into run_cluster_trial(), no separate CLI flag needed
here for it.

Usage (on a login node, e.g. cm4login1/2):
    ./venv/bin/python3 -m rtv_solver.pipeline.srl_sweep_coordinator <sweep_id> <count>

2026-09-16: SBATCH_SCRIPT is now overridable via the SBATCH_SCRIPT env var
(see chat) - lets a second coordinator instance submit trials to a
different partition/QOS (e.g. submit_srl_cluster_trial_serial_std.sbatch on
serial_std/cm4_serial) to spread sweep trials across separate memory quota
pools instead of all queueing behind the same serial_long QOS:
    SBATCH_SCRIPT=submit_srl_cluster_trial_serial_std.sbatch \\
        ./venv/bin/python3 -m rtv_solver.pipeline.srl_sweep_coordinator <sweep_id> <count>
Default (unset) keeps the original serial_long behavior unchanged.
"""
import json
import os
import subprocess
import sys
import time
import uuid

import wandb

SBATCH_SCRIPT = os.environ.get("SBATCH_SCRIPT", "submit_srl_cluster_trial.sbatch")
POLL_INTERVAL_SECONDS = 120


def submit_job(run_id: str, reward_mode: str, actor_lr: float, critic_lr: float) -> str:
    result = subprocess.run(
        ["sbatch", "-M", "serial", f"--export=RUN_ID={run_id},REWARD_MODE={reward_mode},ACTOR_LR={actor_lr},CRITIC_LR={critic_lr}", SBATCH_SCRIPT],
        capture_output=True, text=True, check=True,
    )
    job_id = result.stdout.strip().split()[3]
    print(f"[coordinator] submitted job {job_id} for run_id={run_id} reward_mode={reward_mode} actor_lr={actor_lr:.6f} critic_lr={critic_lr:.6f}")
    return job_id


def wait_for_job(job_id: str) -> None:
    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        result = subprocess.run(
            ["squeue", "-M", "serial", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True,
        )
        state = result.stdout.strip()
        if not state:
            print(f"[coordinator] job {job_id} no longer in queue - assuming finished")
            return
        print(f"[coordinator] job {job_id} state={state}, still waiting...")


def run_cluster_trial() -> None:
    wandb.init()
    config = wandb.config
    run_id = f"cluster_{uuid.uuid4().hex[:8]}"

    job_id = submit_job(run_id, config.reward_mode, config.actor_lr, config.critic_lr)
    wait_for_job(job_id)

    result_path = f"outputs/srl_training_sweep/{run_id}/result.json"
    try:
        with open(result_path) as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"[coordinator] {result_path} missing - job {job_id} likely failed/timed out, logging failure")
        wandb.log({"best_val_service_rate": 0.0, "test_service_rate": 0.0, "trial_failed": True})
        wandb.finish(exit_code=1)
        return

    wandb.log({
        "best_epoch": result["best_epoch"],
        "best_val_service_rate": result["best_val_service_rate"],
        "test_service_rate": result["test_service_rate"],
    })
    for point in result["val_curve"]:
        wandb.log({"val_service_rate_curve": point["service_rate"], "epoch": point["epoch"]})
    print(f"[coordinator] run_id={run_id} DONE, test_service_rate={result['test_service_rate']:.4f}")
    wandb.finish()


if __name__ == "__main__":
    sweep_id = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    wandb.agent(sweep_id, function=run_cluster_trial, count=count)
