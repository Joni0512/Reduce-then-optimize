"""
2026-09-11: cluster-side coordinator for the GAT+TD-bootstrap+bufferfix
wandb sweep - see chat. Lets the LRZ cluster share the SAME sweep
(jonathan-homafar-technical-university-of-munich/srl-td-bootstrap-gat-
bufferfix-sweep/4nwwa43u) the Mac's local agent is already pulling from,
without needing internet on a compute node:

- THIS process runs on a LOGIN node (has internet) via wandb.agent(...,
  function=...) - wandb.agent handles the sweep-server communication
  (fetching the next gamma/tau suggestion, reporting the result) itself,
  calling our `run_cluster_trial` once per trial.
- `run_cluster_trial` does NOT do the heavy computation in-process. It
  submits sweep_td_bootstrap_gat_bufferfix_cluster_trial.py as a normal
  sbatch job (which never touches wandb), polls squeue until it finishes,
  reads back that job's result.json, and logs the metric to wandb itself.
- Net effect: the actual 12-instance training runs on a compute node (no
  internet needed there), while wandb communication happens only on the
  login node - respects both the compute-node network restriction and
  login-node etiquette (this process is I/O-bound, not compute-heavy).

Usage (on a login node, e.g. cm4login1/2):
    ./venv/bin/python3 -m rtv_solver.pipeline.sweep_td_bootstrap_gat_bufferfix_coordinator <count>
"""
import json
import subprocess
import sys
import time
import uuid

import wandb

REPO_ROOT_ON_CLUSTER = "~/Reduce-then-optimize"  # subprocess runs with cwd=REPO_ROOT_ON_CLUSTER via shell
SWEEP_ID = "jonathan-homafar-technical-university-of-munich/srl-td-bootstrap-gat-bufferfix-sweep/4nwwa43u"
SBATCH_SCRIPT = "submit_sweep_td_bootstrap_gat_bufferfix_cluster_trial.sbatch"
POLL_INTERVAL_SECONDS = 120


def submit_job(run_id: str, gamma: float, tau: float) -> str:
    result = subprocess.run(
        ["sbatch", "-M", "serial", f"--export=RUN_ID={run_id},GAMMA={gamma},TAU={tau}", SBATCH_SCRIPT],
        capture_output=True, text=True, check=True,
    )
    # sbatch prints e.g. "Submitted batch job 5488999 on cluster serial"
    job_id = result.stdout.strip().split()[3]
    print(f"[coordinator] submitted job {job_id} for run_id={run_id} gamma={gamma:.4f} tau={tau:.4f}")
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

    job_id = submit_job(run_id, config.gamma, config.tau)
    wait_for_job(job_id)

    result_path = f"outputs/sweep_td_bootstrap_gat_bufferfix_cluster/{run_id}/result.json"
    try:
        with open(result_path) as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"[coordinator] {result_path} missing - job {job_id} likely failed/timed out, logging failure")
        wandb.log({"mean_final_service_rate": 0.0, "trial_failed": True})
        wandb.finish(exit_code=1)
        return

    for instance, rate in result["final_service_rates"].items():
        wandb.log({f"final_service_rate_{instance}": rate})
    wandb.log({"mean_final_service_rate": result["mean_final_service_rate"]})
    print(f"[coordinator] run_id={run_id} DONE, mean_final_service_rate={result['mean_final_service_rate']:.4f}")
    wandb.finish()


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    wandb.agent(SWEEP_ID, function=run_cluster_trial, count=count)
