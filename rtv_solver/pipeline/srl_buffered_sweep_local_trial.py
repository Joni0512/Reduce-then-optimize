"""
2026-10-07: local wandb-agent program for the buffered-actor sweep (sweep_srl_buffered.yaml). Runs one trial
IN-PROCESS on this machine with the sweep's (actor_lr, critic_lr, seed); everything else is fixed to the
protocol of srl_buffered_actor_lr_trial.py (buffered actor update, batch 16, 10 pretrain + EPOCHS main epochs,
pickup_slack, twin critic, no warmstart). Cluster trials go through srl_buffered_sweep_coordinator.py
instead (compute nodes have no internet); both feed the same wandb sweep.
"""
import traceback
import uuid

import wandb

# import first: this module re-patches the FeatureBuilder pickup_slack flags AFTER srl_training_loop's own patch
from rtv_solver.pipeline import srl_buffered_actor_lr_trial as trial
from rtv_solver.pipeline.srl_training_loop import REPO_ROOT
from rtv_solver.pipeline.srl_buffered_sweep_common import read_trial_outcome, log_outcome

EPOCHS = 20  # 10 pretrain + 20 main epochs = the 24 h-compatible protocol of the cm4 scan


def main() -> None:
    wandb.init()
    config = wandb.config
    run_id = f"bufsweep_local_{uuid.uuid4().hex[:8]}"
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== srl_buffered_sweep_local_trial {run_id}: actor_lr={config.actor_lr} critic_lr={config.critic_lr} seed={config.seed} epochs={EPOCHS} ===")
    crashed = False
    try:
        trial._run(run_id, float(config.actor_lr), float(config.critic_lr), int(config.seed), EPOCHS, output_dir)
    except Exception:
        crashed = True
        with open(output_dir / "crash_traceback.txt", "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_buffered_sweep_local_trial {run_id} FAILED - traceback in {output_dir / 'crash_traceback.txt'}")
    status, payload = read_trial_outcome(output_dir)
    log_outcome(wandb, status, payload, extra={"run_id": run_id, "where": "local"})
    wandb.finish(exit_code=1 if (crashed or status != "done") else 0)


if __name__ == "__main__":
    main()
