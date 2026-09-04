"""
2026-09-04: fine-grained target_critic Polyak tau grid [0.002, 0.003, 0.004],
full balanced 12-instance set, seed=42 - see chat. Narrows the gap between
tau=0.001 (final=0.421, worse) and tau=0.005 (final=0.550, best result so
far) - checking whether there's a clean local optimum around tau=0.005, or
a wider plateau, or a sharp peak.

Critic pretrained ONCE (shared across all tau values) - only the
target_critic Polyak blending during the 20 SRL episodes differs per tau.
"""
import copy

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

TAU_VALUES = [0.002, 0.003, 0.004]

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
