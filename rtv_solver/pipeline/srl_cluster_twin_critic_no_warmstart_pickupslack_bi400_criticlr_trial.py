"""
Seed-fixed (42), critic_lr-parametrized cluster variant of
srl_local_twin_critic_no_warmstart_pickupslack_bi400_trial.py (critic_lr=0.0010833586558285635
baseline at bi400/ss100: best_epoch=30, best_val=0.6331, test=0.6233 - clearly worse than the
bi200/ss100 counterpart at the same critic_lr, val=0.7215/test=0.7702, matching RHO's own known
bi400-worse-than-bi200 effect). This scans critic_lr at bi400/ss100, same two values (0.0015,
0.002) already used in the bi200/ss100 scan, to see if bi400 has a different "sweet spot" than
bi200 or just a uniformly lower ceiling. See chat (2026-10-02).

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_bi400_criticlr_trial <critic_lr>
"""
import sys
import traceback

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

EPOCHS = 30
VAL_EVERY_N_EPOCHS = 5
CRITIC_PRETRAIN_EPOCHS = 10
GAMMA = 0.99
TAU = 0.005
BATCH_INTERVAL = 400
STEP_SIZE = 100
SEED = 42

REWARD_MODE = "local_positive"
ACTOR_LR = 0.0031923396291543868


def main() -> None:
    critic_lr = float(sys.argv[1])
    critic_lr_tag = str(critic_lr).replace(".", "p")
    run_id = f"twin_critic_no_warmstart_pickupslack_bi400_criticlr{critic_lr_tag}"
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    print(f"=== srl_cluster_twin_critic_no_warmstart_pickupslack_bi400_criticlr_trial {run_id}: reward_mode={REWARD_MODE} actor_lr={ACTOR_LR} critic_lr={critic_lr} tau={TAU} gamma={GAMMA} epochs={EPOCHS} batch_interval={BATCH_INTERVAL} step_size={STEP_SIZE} pickup_slack=True feature_size={_feat_builder_module.FeatureBuilder.FEATURE_SIZE} use_twin_critic=True use_actor_warmstart=False ===")
    try:
        result = run_srl_training_loop(
            reward_mode=REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=critic_lr,
            output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
            gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
            critic_pretrain_epochs=CRITIC_PRETRAIN_EPOCHS,
            batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=SEED,
            use_twin_critic=True, use_actor_warmstart=False,
        )

        config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
        active_feature_builder = select_feature_builder_class(config_template)
        model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
        checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
        print(f"=== srl_cluster_twin_critic_no_warmstart_pickupslack_bi400_criticlr_trial {run_id} DONE: best_epoch={result.best_epoch} best_val_service_rate={result.best_val_service_rate:.4f} test_service_rate={test_service_rate:.4f} ===")
    except Exception as e:
        crash_path = output_dir / "crash_traceback.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_cluster_twin_critic_no_warmstart_pickupslack_bi400_criticlr_trial {run_id} FAILED: {e!r} - full traceback written to {crash_path}")


if __name__ == "__main__":
    main()
