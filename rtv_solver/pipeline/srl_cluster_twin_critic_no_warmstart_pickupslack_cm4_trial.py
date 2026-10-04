"""
cm4-sized (< 24 h) baseline protocol for the OLD per-step SRL architecture, to be compared
against the later replay-buffer/mini-batch variant (see chat, 2026-10-04): twin-critic,
no-warmstart, pickup_slack=True, actor_lr fixed at the baseline value, bi200/ss100, 10 critic
pretrain epochs + a configurable number of main epochs (default 20: measured 19-21 h on
serial_std, vs 24-27.5 h for 30 epochs - see sacct elapsed times in chat). critic_lr, seed and
epochs come from the command line; everything else is identical to
srl_cluster_twin_critic_no_warmstart_pickupslack_seed_trial.py (30 epochs, baseline critic_lr).
The training loop has no LR scheduler, so epoch k of a 20-epoch run is distributed like epoch k of
a 30-epoch run - best val over epochs 5..20 is therefore comparable to the epoch<=20 part of the
existing 30-epoch val curves.

Writes result.json (incl. val_curve/overfit_curve) like srl_cluster_twin_critic_no_warmstart_
sweep_trial.py, run_id encodes epochs/critic_lr/seed so repeated runs never share an output dir.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial <critic_lr> <seed> [epochs=20]
"""
import json
import sys
import traceback
from pathlib import Path

import torch

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

# Re-patch AFTER srl_training_loop's own import-time patch (which forces False) - see
# srl_local_twin_critic_no_warmstart_pickupslack_trial.py's docstring for the full rationale.
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = True
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
    + (_feat_builder_module.FeatureBuilder._PICKUP_SLACK_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE else 0)
)

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

VAL_EVERY_N_EPOCHS = 5
CRITIC_PRETRAIN_EPOCHS = 10
GAMMA = 0.99
TAU = 0.005
BATCH_INTERVAL = 200
STEP_SIZE = 100
REWARD_MODE = "local_positive"
ACTOR_LR = 0.0031923396291543868  # baseline, unchanged


def main() -> None:
    critic_lr = float(sys.argv[1])
    seed = int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    critic_lr_tag = str(critic_lr).replace(".", "p")
    run_id = f"cm4_ep{epochs}_ps_clr{critic_lr_tag}_s{seed}"
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial {run_id}: actor_lr={ACTOR_LR} critic_lr={critic_lr} seed={seed} epochs={epochs} (+{CRITIC_PRETRAIN_EPOCHS} critic pretrain) batch_interval={BATCH_INTERVAL} step_size={STEP_SIZE} pickup_slack=True feature_size={_feat_builder_module.FeatureBuilder.FEATURE_SIZE} ===")
    try:
        _run(run_id, critic_lr, seed, epochs, output_dir)
    except Exception:
        crash_path = output_dir / "crash_traceback.txt"
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial {run_id} FAILED - full traceback written to {crash_path}")
        raise


def _run(run_id: str, critic_lr: float, seed: int, epochs: int, output_dir: Path) -> None:
    result = run_srl_training_loop(
        reward_mode=REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=critic_lr,
        output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
        gamma=GAMMA, tau=TAU, epochs=epochs, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
        critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
        batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=seed,
        use_twin_critic=True, use_actor_warmstart=False,
    )

    config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=seed)
    active_feature_builder = select_feature_builder_class(config_template)
    model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")

    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump({
            "run_id": run_id, "actor_lr": ACTOR_LR, "critic_lr": critic_lr, "seed": seed,
            "epochs": epochs, "critic_pretrain_epochs": CRITIC_PRETRAIN_EPOCHS,
            "batch_interval": BATCH_INTERVAL, "step_size": STEP_SIZE, "pickup_slack": True,
            "best_epoch": result.best_epoch,
            "best_val_service_rate": result.best_val_service_rate,
            "test_service_rate": test_service_rate,
            "val_curve": result.val_curve,
            "overfit_curve": result.overfit_curve,
        }, f, indent=2)
    print(f"=== srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial {run_id} DONE: best_epoch={result.best_epoch} best_val_service_rate={result.best_val_service_rate:.4f} test_service_rate={test_service_rate:.4f} - wrote {result_path} ===")


if __name__ == "__main__":
    main()
