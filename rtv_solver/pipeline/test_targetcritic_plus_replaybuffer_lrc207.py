"""
2026-08-31: first-ever combined test of target_critic (Option B, hard periodic
copy - see chat) AND the replay buffer together, on lrc207 - see chat. Both
mechanisms are already independent, unguarded params in
train_srl_single_instance.train() (only checked against freeze_critic, never
against each other), so this needs no code change - just passing both.

target_critic_update_interval=5 - X=5 was the best-performing interval of the
X=1/5/10 tested earlier this session (all on the OLD unbalanced split,
without the replay buffer or gradient clipping that exist now).
Replay-buffer at default HPs (sigma=1.0, critic_lr=1e-3, capacity=40) -
same defaults as run_srl_balanced_replaybuffer_12instances.py.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print(f"Running {INSTANCE} with target_critic (X=5) + replay buffer combined...")

    train(
        instance=INSTANCE,
        episodes=EPISODES,
        batch_interval=BATCH_INTERVAL,
        step_size=STEP_SIZE,
        seed=SEED,
        actor_checkpoint=ACTOR_CHECKPOINT,
        freeze_critic=False,
        shared_critic=shared_critic,
        label_suffix="_balanced_targetcritic_x5_replaybuffer",
        target_critic_update_interval=5,
        use_replay_buffer=True,
        replay_capacity=40,
        replay_batch_size=12,
        replay_update_group_size=3,
    )

    print(f"\n=== {INSTANCE} (target_critic X=5 + replay buffer) DONE ===")
