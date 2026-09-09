"""
2026-09-09: TD-bootstrap critic target (critic_target_mode="td_bootstrap"),
gamma=0.99, target_critic Polyak tau=0.005, on the full balanced 12-instance
test set - see chat. Local single-seed (42) run, same config as the earlier
single-instance TD-bootstrap tests (lrc207/lc108) that showed an
instance-specific collapse on lrc207 but stable results on lc108 - checking
whether that pattern holds set-wide.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances with TD-bootstrap critic target (gamma=0.99, tau=0.005) + replay buffer.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, td_bootstrap gamma=0.99 tau=0.005 + replay buffer) ===")
        try:
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                actor_checkpoint=ACTOR_CHECKPOINT,
                freeze_critic=False,
                shared_critic=shared_critic,
                label_suffix="_balanced_td_bootstrap_gamma0.99_tau0.005",
                critic_target_mode="td_bootstrap",
                gamma=0.99,
                target_critic_polyak_tau=0.005,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, td_bootstrap gamma=0.99 tau=0.005) - failed: {failed} ===")
