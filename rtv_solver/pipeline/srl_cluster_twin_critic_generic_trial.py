"""
2026-09-25: generic (parametrized) twin-critic cluster trial - see chat. Replaces one-off
scripts like srl_cluster_twin_critic_trial.py/srl_cluster_twin_critic_tau0001_trial.py for
running MULTIPLE twin-critic tests at different actor_lr/critic_lr combinations (picked from
the best trials found so far in the single-critic srl-training-loop-sweep, see
figures_export/SRL/sweep_all_finished_runs_table.png) without duplicating a near-identical
file per combination. tau/gamma/reward_mode/seed are also CLI args so this covers the
tau=0.001 vs. tau=0.005 comparison too.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_cluster_twin_critic_generic_trial <run_id> <reward_mode> <actor_lr> <critic_lr> <tau> <seed>
"""
import sys
import traceback

import torch

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5
GAMMA = 0.99
BATCH_INTERVAL = 200
STEP_SIZE = 100


def main(run_id: str, reward_mode: str, actor_lr: float, critic_lr: float, tau: float, seed: int) -> None:
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    print(f"=== srl_cluster_twin_critic_generic_trial {run_id}: reward_mode={reward_mode} actor_lr={actor_lr} critic_lr={critic_lr} tau={tau} gamma={GAMMA} seed={seed} use_twin_critic=True ===")
    try:
        result = run_srl_training_loop(
            reward_mode=reward_mode, actor_lr=actor_lr, critic_lr=critic_lr,
            output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
            gamma=GAMMA, tau=tau, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
            batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=seed,
            use_twin_critic=True,
        )

        config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=seed)
        active_feature_builder = select_feature_builder_class(config_template)
        model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
        checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
        print(f"=== srl_cluster_twin_critic_generic_trial {run_id} DONE: best_epoch={result.best_epoch} best_val_service_rate={result.best_val_service_rate:.4f} test_service_rate={test_service_rate:.4f} ===")
    except Exception as e:
        crash_path = output_dir / "crash_traceback.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_cluster_twin_critic_generic_trial {run_id} FAILED: {e!r} - full traceback written to {crash_path}")


if __name__ == "__main__":
    run_id = sys.argv[1]
    reward_mode = sys.argv[2]
    actor_lr = float(sys.argv[3])
    critic_lr = float(sys.argv[4])
    tau = float(sys.argv[5])
    seed = int(sys.argv[6])
    main(run_id, reward_mode, actor_lr, critic_lr, tau, seed)
