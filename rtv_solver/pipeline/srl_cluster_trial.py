"""
2026-09-13: cluster-side trial for the SIL/SRL(local)/SRL(local_positive)
actor_lr x critic_lr wandb random-search sweep (see chat/srl deck) - mirrors
sweep_td_bootstrap_gat_bufferfix_cluster_trial.py's pattern: compute nodes
have no internet, so this script does NOT touch wandb at all. It runs
srl_training_loop.run_srl_training_loop() (train on 38, validate on 9 every
val_every_n_epochs, best-val checkpoint reloaded), then evaluates that best
checkpoint on the 9 held-out TEST_INSTANCES, and writes ONE result.json - a
coordinator process on the LOGIN node (which has internet) reads it back and
reports it to wandb.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_cluster_trial <run_id> <reward_mode> <actor_lr> <critic_lr>
"""
import json
import sys
import traceback
from pathlib import Path

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.structure.config import Config

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5
GAMMA = 0.99
TAU = 0.001
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42


def main() -> None:
    run_id = sys.argv[1]
    reward_mode = sys.argv[2]
    actor_lr = float(sys.argv[3])
    critic_lr = float(sys.argv[4])
    print(f"=== srl_cluster_trial {run_id}: reward_mode={reward_mode} actor_lr={actor_lr} critic_lr={critic_lr} ===")

    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run(run_id, reward_mode, actor_lr, critic_lr, output_dir)
    except Exception:
        # 2026-09-15: dump the FULL traceback to its own file, independent of
        # whatever happened to stdout/stderr/logging at crash time - see
        # chat (a run died with an empty, unreadable "Logging error" and no
        # visible traceback in the sbatch .err file; root cause still
        # unconfirmed). Re-raise after so the process still exits non-zero
        # and SLURM/wandb see it as failed, same as before.
        crash_path = output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_cluster_trial {run_id} CRASHED - full traceback written to {crash_path}")
        raise


def _run(run_id: str, reward_mode: str, actor_lr: float, critic_lr: float, output_dir: Path) -> None:
    result = run_srl_training_loop(
        reward_mode=reward_mode, actor_lr=actor_lr, critic_lr=critic_lr,
        output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
        gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
        batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=SEED,
    )

    config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
    import torch
    from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
    from rtv_solver.pipeline import select_feature_builder_class
    active_feature_builder = select_feature_builder_class(config_template)
    model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
    print(f"=== srl_cluster_trial {run_id}: test service rate (best_epoch={result.best_epoch}) = {test_service_rate:.4f} ===")

    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump({
            "run_id": run_id, "reward_mode": reward_mode, "actor_lr": actor_lr, "critic_lr": critic_lr,
            "best_epoch": result.best_epoch,
            "best_val_service_rate": result.best_val_service_rate,
            "test_service_rate": test_service_rate,
            "val_curve": result.val_curve,
            "overfit_curve": result.overfit_curve,
        }, f, indent=2)
    print(f"=== srl_cluster_trial {run_id} DONE - wrote {result_path} ===")


if __name__ == "__main__":
    main()
