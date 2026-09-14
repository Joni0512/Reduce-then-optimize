"""
2026-09-14: local-Mac counterpart to srl_sweep_coordinator.py/srl_cluster_trial.py
- see chat. Runs a trial IN-PROCESS (no sbatch, no cluster) directly against
the SAME wandb sweep the cluster coordinators pull from, so local and
cluster runs share one sweep's trial budget. Useful both for speed (extra
parallel trial) and as a diagnostic: if local trials finish cleanly while
cluster trials keep failing on the LRZ Gurobi token-server error, that
isolates the failure to the LRZ side rather than this code.

Only run ONE of these locally at a time - the personal Gurobi WLS license
has a baseline of 2 concurrent sessions (see CLAUDE.md's Cluster Gurobi
License Fix note); this script alone should stay within that, but don't
also start a second local trial or other local Gurobi-heavy run at the
same time.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_local_trial <sweep_id> [count]
"""
import sys
import uuid

import torch
import wandb

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5
GAMMA = 0.99
TAU = 0.001
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42


def run_one_trial() -> None:
    wandb.init()
    config = wandb.config
    run_id = f"local_{uuid.uuid4().hex[:8]}"
    reward_mode = config.reward_mode
    actor_lr = config.actor_lr
    critic_lr = config.critic_lr

    print(f"=== srl_local_trial {run_id}: reward_mode={reward_mode} actor_lr={actor_lr} critic_lr={critic_lr} ===")
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    try:
        result = run_srl_training_loop(
            reward_mode=reward_mode, actor_lr=actor_lr, critic_lr=critic_lr,
            output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
            gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
            batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=SEED,
        )

        config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
        active_feature_builder = select_feature_builder_class(config_template)
        model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
        checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
        wandb.log({
            "best_epoch": result.best_epoch,
            "best_val_service_rate": result.best_val_service_rate,
            "test_service_rate": test_service_rate,
        })
        print(f"=== srl_local_trial {run_id} DONE: test_service_rate={test_service_rate:.4f} ===")
        wandb.finish()
    except Exception as e:
        print(f"!!! srl_local_trial {run_id} FAILED: {e!r}")
        wandb.log({"best_val_service_rate": 0.0, "test_service_rate": 0.0, "trial_failed": True})
        wandb.finish(exit_code=1)


if __name__ == "__main__":
    sweep_id = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    wandb.agent(sweep_id, function=run_one_trial, count=count)
