"""
2026-08-31: grid over target_critic Polyak tau in [0.1, 0.2, 0.3, 0.4, 0.5],
same seed=42, on the full balanced 12-instance test set - see chat.
tau=0.5 (the first, deliberately simple test) showed no improvement over
hard-copy X=5 on lrc207 - checking whether a smaller, more typical tau
(closer to DDPG/TD3's ~0.005-0.05 range, though still much larger here)
behaves differently, and whether that holds across the full set.

Critic pretrained ONCE (shared across all tau values, same as other balanced
sweeps) - only the target_critic Polyak blending during the 20 SRL episodes
differs per tau.
"""
import copy

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

TAU_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    base_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances for each tau value.")

    failed = []
    for tau in TAU_VALUES:
        for instance in TEST_INSTANCES:
            print(f"\n=== tau={tau} instance={instance} (balanced, target_critic Polyak + replay buffer) ===")
            try:
                critic = copy.deepcopy(base_critic)
                train(
                    instance=instance,
                    episodes=EPISODES,
                    batch_interval=BATCH_INTERVAL,
                    step_size=STEP_SIZE,
                    seed=SEED,
                    actor_checkpoint=ACTOR_CHECKPOINT,
                    freeze_critic=False,
                    shared_critic=critic,
                    label_suffix=f"_balanced_targetcritic_polyak{tau}_replaybuffer",
                    target_critic_polyak_tau=tau,
                    use_replay_buffer=True,
                    replay_capacity=40,
                    replay_batch_size=12,
                    replay_update_group_size=3,
                )
            except Exception as e:
                print(f"!!! tau={tau} instance={instance} FAILED: {e!r} - skipping, continuing.")
                failed.append((tau, instance))

    print(f"\n=== ALL TAU VALUES x 12 INSTANCES DONE - failed: {failed} ===")
