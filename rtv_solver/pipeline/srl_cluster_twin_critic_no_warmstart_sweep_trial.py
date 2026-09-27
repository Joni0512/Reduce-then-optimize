"""
Cluster-side trial for the twin-critic, no-warmstart actor_lr x critic_lr x seed wandb
random-search sweep - same pattern as srl_cluster_trial.py (compute nodes have no internet, so
this script never touches wandb; a login-node coordinator, srl_twin_critic_no_warmstart_sweep_
coordinator.py, submits this as an sbatch job per trial and reports the result.json back to
wandb). Fixed: use_twin_critic=True, use_actor_warmstart=False, epochs=30, tau=0.005,
reward_mode=local_positive (see chat - the best known twin-critic config, only actor_lr/
critic_lr/seed swept here).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_sweep_trial <run_id> <actor_lr> <critic_lr> <seed>
"""
import json
import sys
import traceback
from pathlib import Path

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.structure.config import Config

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 30
VAL_EVERY_N_EPOCHS = 5
CRITIC_PRETRAIN_EPOCHS = 10
GAMMA = 0.99
TAU = 0.005
BATCH_INTERVAL = 200
STEP_SIZE = 100
REWARD_MODE = "local_positive"


def main() -> None:
    run_id = sys.argv[1]
    actor_lr = float(sys.argv[2])
    critic_lr = float(sys.argv[3])
    seed = int(sys.argv[4])
    print(f"=== srl_cluster_twin_critic_no_warmstart_sweep_trial {run_id}: actor_lr={actor_lr} critic_lr={critic_lr} seed={seed} ===")

    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run(run_id, actor_lr, critic_lr, seed, output_dir)
    except Exception:
        crash_path = output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_cluster_twin_critic_no_warmstart_sweep_trial {run_id} CRASHED - full traceback written to {crash_path}")
        raise


def _run(run_id: str, actor_lr: float, critic_lr: float, seed: int, output_dir: Path) -> None:
    result = run_srl_training_loop(
        reward_mode=REWARD_MODE, actor_lr=actor_lr, critic_lr=critic_lr,
        output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
        gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
        critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
        batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=seed,
        use_twin_critic=True, use_actor_warmstart=False,
    )

    from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
    from rtv_solver.pipeline import select_feature_builder_class
    import torch
    config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=seed)
    active_feature_builder = select_feature_builder_class(config_template)
    model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
    print(f"=== srl_cluster_twin_critic_no_warmstart_sweep_trial {run_id}: test service rate (best_epoch={result.best_epoch}) = {test_service_rate:.4f} ===")

    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump({
            "run_id": run_id, "actor_lr": actor_lr, "critic_lr": critic_lr, "seed": seed,
            "best_epoch": result.best_epoch,
            "best_val_service_rate": result.best_val_service_rate,
            "test_service_rate": test_service_rate,
            "val_curve": result.val_curve,
            "overfit_curve": result.overfit_curve,
        }, f, indent=2)
    print(f"=== srl_cluster_twin_critic_no_warmstart_sweep_trial {run_id} DONE - wrote {result_path} ===")


if __name__ == "__main__":
    main()
