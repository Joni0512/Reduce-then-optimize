"""
2026-09-09: first local correctness test for TD-bootstrap (critic_target_mode
="td_bootstrap") - see chat and docs/SRL_Design.md's TD Bootstrap plan
section. Deliberately gcn critic (not gat) + MLP actor (both defaults) so
this runs quickly locally, on ONE instance (lrc207, our recurring test case
this session) instead of all 12 - a first look at whether TD-bootstrap
trains sensibly at all before committing to a bigger multi-instance/seed run.

Baseline hyperparameters throughout (sigma=1.0, critic_lr=1e-3,
actor_lr=1e-4, replay_capacity=40, max_cardinality=2 - all train() defaults,
unchanged), same as every other baseline comparison this session.
target_critic is built via Polyak averaging (tau=0.005, our best-performing
tau from the target_critic experiments earlier this session) - required
since td_bootstrap raises ValueError without an explicit target_critic.
gamma=0.99 (train()'s new default).

train_srl_single_instance.py's plotting now includes a 5th panel (mean
critic vs. target_critic Q-value per episode) specifically to inspect this
kind of run.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED, pretrain_shared_critic,
)
from rtv_solver.pipeline.train_srl_single_instance import train

TEST_INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (gcn, standard Monte Carlo pretrain - unchanged)...")
    shared_critic = pretrain_shared_critic()
    print(f"Pretraining done - running {TEST_INSTANCE} with critic_target_mode=td_bootstrap, gamma=0.99.")

    train(
        instance=TEST_INSTANCE,
        episodes=EPISODES,
        batch_interval=BATCH_INTERVAL,
        step_size=STEP_SIZE,
        seed=SEED,
        actor_checkpoint=ACTOR_CHECKPOINT,
        freeze_critic=False,
        shared_critic=shared_critic,
        label_suffix="_td_bootstrap_test_gcn",
        use_replay_buffer=True,
        replay_capacity=40,
        replay_batch_size=12,
        replay_update_group_size=3,
        target_critic_polyak_tau=0.005,
        critic_target_mode="td_bootstrap",
        gamma=0.99,
    )
    print(f"\n=== {TEST_INSTANCE} (td_bootstrap, gcn, gamma=0.99) DONE ===")
