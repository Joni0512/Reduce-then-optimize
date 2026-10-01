"""
One-off: repeated 5-epoch attempts of the "stable" critic_lr=0.0010833586558285635 config
(pickup_slack=True, seed=42, no-warmstart) with weight-norm-per-step logging, since the first
attempt (srl_weight_norm_divergence_trial.py) collapsed too (best_val=0.0000) despite being the
same config that earlier reached best_val=0.7215 over 30 epochs - see chat (2026-10-01). Keeps
retrying until a non-collapsed (val > 0.3 at epoch 5, somewhat arbitrary threshold) attempt is
found or MAX_ATTEMPTS is reached, so we have at least one "good" weight-norm trajectory to
compare against the known collapsing critic_lr=0.001 trajectory.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_weight_norm_stable_retry_trial
"""
import json
import traceback

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT

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
CRITIC_LR = 0.0010833586558285635

MAX_ATTEMPTS = 5
COLLAPSE_THRESHOLD = 0.3


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / "weight_norm_divergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        tag = f"stable_criticlr0p00108_attempt{attempt}"
        run_out_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / f"weight_norm_divergence_{tag}"
        trace: list[dict] = []
        print(f"=== srl_weight_norm_stable_retry_trial attempt {attempt}/{MAX_ATTEMPTS} ===")
        try:
            result = run_srl_training_loop(
                reward_mode=REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
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
                    "tag": tag, "attempt": attempt, "critic_lr": CRITIC_LR, "actor_lr": ACTOR_LR,
                    "best_epoch": result.best_epoch, "best_val_service_rate": result.best_val_service_rate,
                    "trace": trace,
                }, f)
            status = "GOOD (non-collapsed)" if result.best_val_service_rate > COLLAPSE_THRESHOLD else "collapsed"
            print(f"=== srl_weight_norm_stable_retry_trial attempt {attempt} DONE: best_epoch={result.best_epoch} best_val={result.best_val_service_rate:.4f} [{status}] -> wrote {out_path} ===")
            if result.best_val_service_rate > COLLAPSE_THRESHOLD:
                print(f"=== Found a non-collapsed attempt ({attempt}) - stopping retries. ===")
                break
        except Exception as e:
            crash_path = run_out_dir / "crash_traceback.txt"
            run_out_dir.mkdir(parents=True, exist_ok=True)
            with open(crash_path, "w") as f:
                traceback.print_exc(file=f)
            out_path = out_dir / f"{tag}_partial.json"
            with open(out_path, "w") as f:
                json.dump({"tag": tag, "attempt": attempt, "critic_lr": CRITIC_LR, "actor_lr": ACTOR_LR, "trace": trace}, f)
            print(f"!!! srl_weight_norm_stable_retry_trial attempt {attempt} FAILED: {e!r} - traceback @ {crash_path}, partial trace @ {out_path}")


if __name__ == "__main__":
    main()
