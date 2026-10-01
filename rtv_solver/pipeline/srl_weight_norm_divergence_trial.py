"""
One-off: short (5-epoch) side-by-side run of the stable critic_lr=0.0010833586558285635
(baseline) config vs. the collapsing critic_lr=0.001 config (both no-warmstart, twin-critic,
pickup_slack=True, seed=42 - see twin_critic_no_warmstart_pickupslack_1 (best_val=0.7215) and
twin_critic_no_warmstart_pickupslack_criticlr001_1 (collapsed, best_val=0.0086) for the full
30-epoch results), this time logging the actor's weight L2 norm after every single
rolling-horizon gradient step (coaml_pipeline.py's weight_norm_history) via
run_srl_training_loop's new weight_norm_trace parameter. Only 5 epochs - the collapse was
already visible at epoch 5 (val=0.0000) in the full run, so this is enough to find the
divergence point without re-running all 30 epochs.

Writes raw per-step traces to JSON for later plotting - does not plot itself.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_weight_norm_divergence_trial
"""
import json
import traceback

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT

# Re-patch AFTER srl_training_loop's own import-time patch (which forces False) - matches
# the pickup_slack trials this comparison is drawn from.
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = True
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
    + (_feat_builder_module.FeatureBuilder._PICKUP_SLACK_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE else 0)
)

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 5
VAL_EVERY_N_EPOCHS = 5
CRITIC_PRETRAIN_EPOCHS = 10
GAMMA = 0.99
TAU = 0.005
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42

REWARD_MODE = "local_positive"
ACTOR_LR = 0.0031923396291543868

CONFIGS = {
    "stable_criticlr0p00108": 0.0010833586558285635,
    "collapse_criticlr0p001": 0.001,
}


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / "weight_norm_divergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, critic_lr in CONFIGS.items():
        run_id = f"weight_norm_divergence_{tag}"
        run_out_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
        trace: list[dict] = []
        print(f"=== srl_weight_norm_divergence_trial {tag}: critic_lr={critic_lr} epochs={EPOCHS} ===")
        try:
            result = run_srl_training_loop(
                reward_mode=REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=critic_lr,
                output_dir=run_out_dir, actor_checkpoint=ACTOR_CHECKPOINT,
                gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
                critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
                batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=SEED,
                use_twin_critic=True, use_actor_warmstart=False,
                weight_norm_trace=trace,
            )
            out_path = out_dir / f"{tag}.json"
            with open(out_path, "w") as f:
                json.dump({
                    "tag": tag, "critic_lr": critic_lr, "actor_lr": ACTOR_LR,
                    "best_epoch": result.best_epoch, "best_val_service_rate": result.best_val_service_rate,
                    "trace": trace,
                }, f)
            print(f"=== srl_weight_norm_divergence_trial {tag} DONE: best_epoch={result.best_epoch} best_val={result.best_val_service_rate:.4f} -> wrote {out_path} ===")
        except Exception as e:
            crash_path = run_out_dir / "crash_traceback.txt"
            run_out_dir.mkdir(parents=True, exist_ok=True)
            with open(crash_path, "w") as f:
                traceback.print_exc(file=f)
            # still dump whatever trace we collected before the crash
            out_path = out_dir / f"{tag}_partial.json"
            with open(out_path, "w") as f:
                json.dump({"tag": tag, "critic_lr": critic_lr, "actor_lr": ACTOR_LR, "trace": trace}, f)
            print(f"!!! srl_weight_norm_divergence_trial {tag} FAILED: {e!r} - traceback @ {crash_path}, partial trace @ {out_path}")


if __name__ == "__main__":
    main()
