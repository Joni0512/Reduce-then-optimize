"""
2026-09-09: follow-up TD-bootstrap tests - see chat. First run
(test_srl_td_bootstrap_single_instance.py, lrc207, gamma=0.99, tau=0.005)
collapsed steadily to service_rate=0.0 by episode 7, with mean_critic_q
drifting from -3.07 to -8.09 while target_critic stayed near -3.0 (Polyak
tau=0.005 barely moves it). Testing whether a less aggressive discount
(gamma=0.95) and a much more responsive target_critic (tau=0.1, closer to
the earlier-tested 0.1-0.5 range instead of the very small 0.005) helps -
plus a second instance (lc108, a historically stable/high-service-rate
case, unlike lrc207's chronic instability) to see if the collapse is
instance-specific or a general TD-bootstrap issue.

Runs, in order (single shared critic pretrain, reused for all three):
  1. lrc207, gamma=0.95, tau=0.1
  2. lc108,  gamma=0.99, tau=0.005  (same config as the first test, new instance)
  3. lc108,  gamma=0.95, tau=0.1

gcn critic + MLP actor (both defaults), baseline hyperparameters otherwise,
same as test_srl_td_bootstrap_single_instance.py.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import copy

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED, pretrain_shared_critic,
)
from rtv_solver.pipeline.train_srl_single_instance import train

RUNS = [
    ("lrc207", 0.95, 0.1),
    ("lc108", 0.99, 0.005),
    ("lc108", 0.95, 0.1),
]

if __name__ == "__main__":
    print("Pretraining shared critic (gcn, standard Monte Carlo pretrain - unchanged, reused for all 3 runs)...")
    shared_critic = pretrain_shared_critic()

    for instance, gamma, tau in RUNS:
        label_suffix = f"_td_bootstrap_test_gcn_gamma{gamma}_tau{tau}"
        print(f"\n=== {instance} (td_bootstrap, gcn, gamma={gamma}, tau={tau}) ===")
        try:
            critic = copy.deepcopy(shared_critic)
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                actor_checkpoint=ACTOR_CHECKPOINT,
                freeze_critic=False,
                shared_critic=critic,
                label_suffix=label_suffix,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
                target_critic_polyak_tau=tau,
                critic_target_mode="td_bootstrap",
                gamma=gamma,
            )
        except Exception as e:
            print(f"!!! {instance} (gamma={gamma}, tau={tau}) FAILED: {e!r} - skipping, continuing.")

    print("\n=== ALL TD-BOOTSTRAP FOLLOW-UP RUNS DONE ===")
