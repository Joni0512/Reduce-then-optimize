"""
2026-08-29: pretrains the shared critic ONCE (same TRAIN_INSTANCES/
ACTOR_CHECKPOINT as run_srl_balanced_frozen_12instances.py) and saves its
weights to disk - see chat. Needed so the wandb HP sweep
(sweep_srl_replaybuffer_trial.py) doesn't repeat this expensive step (10
epochs x 12 instances) on every trial: each trial only sweeps the SRL
fine-tuning phase's hyperparameters (replay_capacity, critic_lr, sigma),
which don't affect this pretrain phase (its own critic_lr is fixed at 1e-3
inside pretrain_shared_critic(), unrelated to the swept critic_lr used
later for the per-instance replay-buffer training).
"""
import torch

# 2026-08-29: ACTOR_CHECKPOINT below was SIL-trained BEFORE commit 69b521f
# flipped feat_builder.py's ENABLE_PICKUP_SLACK_FEATURE to True (see chat) -
# that raised v1's FEATURE_SIZE from 84 to 85, breaking checkpoint loading
# for every SRL run using this older checkpoint. Monkey-patched back to False
# here (BEFORE importing anything that builds a FeatureBuilder/model) so this
# script stays comparable to last night's SRL sweeps, which all ran under
# FEATURE_SIZE=84. This is a temporary, local-only override - the user wants
# to first check whether the pickup-slack feature helps before deciding
# whether to retrain the SRL actor checkpoint with it enabled.
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import pretrain_shared_critic, REPO_ROOT

OUTPUT_PATH = REPO_ROOT / "outputs" / "srl_sweep" / "shared_critic_pretrained.pt"

if __name__ == "__main__":
    print("Pretraining shared critic...")
    critic = pretrain_shared_critic()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(critic.state_dict(), OUTPUT_PATH)
    print(f"Saved pretrained critic weights to {OUTPUT_PATH}")
