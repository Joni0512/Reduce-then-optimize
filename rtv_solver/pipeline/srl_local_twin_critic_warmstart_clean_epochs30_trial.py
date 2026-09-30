"""
Same hyperparameters as the no-warmstart baseline (srl_local_twin_critic_no_warmstart_
epochs30_trial.py: actor_lr=0.0031923396291543868, critic_lr=0.0010833586558285635,
tau=0.005, gamma=0.99, epochs=30, seed=42), but use_actor_warmstart=True this time, loading
the NEW, LEAKAGE-FREE SIL checkpoint (sil_training_bi200_ss100_clean_srl_split_legacy_mlp_
seed1, best val=0.7170 at epoch 2 - trained only on 32 of 38 TRAIN_INSTANCES, holding out 6 for
its own validation, never touching VAL_INSTANCES/TEST_INSTANCES - see chat) instead of the old
leaky checkpoint (sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1, which had 4/9
VAL_INSTANCES in its own SIL-train set and 4/9 TEST_INSTANCES in its own SIL-val set).

This gives the first leakage-free warmstart-vs-no-warmstart comparison point at these
hyperparameters: no-warmstart got best_val=0.6872/test=0.7056 at epoch 30 (seed 42); the old
(leaky) warmstart got 0.7085 at epoch 25 - now unknown how much of that gap was leakage vs. a
genuine warmstart benefit.

2026-09-30 correction: the clean SIL checkpoint was actually trained with pickup_slack
enabled (85 features) - the SIL training script never overrode FeatureBuilder's own class
default (ENABLE_PICKUP_SLACK_FEATURE=True in feat_builder.py), so a first run of this script
crashed with a state_dict shape mismatch (checkpoint 85 features vs. the SRL actor's default 84,
since srl_training_loop.py forces pickup_slack=False at its own import time). Re-patching to
True here so the actor's feature set matches what the checkpoint was actually trained with -
this is no longer a "no pickup_slack" baseline warmstart comparison, it's warmstart+pickup_slack
vs. the no-warmstart+pickup_slack run (twin_critic_no_warmstart_pickupslack_1: best_val=0.7215,
test=0.7702 at seed=42).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_local_twin_critic_warmstart_clean_epochs30_trial
"""
import traceback

import torch

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

# Re-patch AFTER srl_training_loop's own import-time patch (which forces False) - see module
# docstring above: the clean checkpoint was trained with pickup_slack enabled.
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = True
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
    + (_feat_builder_module.FeatureBuilder._PICKUP_SLACK_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE else 0)
)

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_clean_srl_split_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 30
VAL_EVERY_N_EPOCHS = 5
CRITIC_PRETRAIN_EPOCHS = 10
GAMMA = 0.99
TAU = 0.005
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42

REWARD_MODE = "local_positive"
ACTOR_LR = 0.0031923396291543868
CRITIC_LR = 0.0010833586558285635  # baseline, unchanged


def main() -> None:
    run_id = "twin_critic_warmstart_clean_pickupslack_epochs30_1"
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    print(f"=== srl_local_twin_critic_warmstart_clean_epochs30_trial {run_id}: reward_mode={REWARD_MODE} actor_lr={ACTOR_LR} critic_lr={CRITIC_LR} tau={TAU} gamma={GAMMA} epochs={EPOCHS} actor_checkpoint={ACTOR_CHECKPOINT} pickup_slack=True feature_size={_feat_builder_module.FeatureBuilder.FEATURE_SIZE} use_twin_critic=True use_actor_warmstart=True ===")
    try:
        result = run_srl_training_loop(
            reward_mode=REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
            output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
            gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
            critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
            batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=SEED,
            use_twin_critic=True, use_actor_warmstart=True,
        )

        config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
        active_feature_builder = select_feature_builder_class(config_template)
        model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
        checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
        print(f"=== srl_local_twin_critic_warmstart_clean_epochs30_trial {run_id} DONE: best_epoch={result.best_epoch} best_val_service_rate={result.best_val_service_rate:.4f} test_service_rate={test_service_rate:.4f} ===")
    except Exception as e:
        crash_path = output_dir / "crash_traceback.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_local_twin_critic_warmstart_clean_epochs30_trial {run_id} FAILED: {e!r} - full traceback written to {crash_path}")


if __name__ == "__main__":
    main()
