"""
2026-09-04: target_critic Polyak with tau=0.005 (classic DDPG/TD3 range),
on the full balanced 12-instance test set - see chat. Single-instance test
on lrc207 showed no improvement over the 0.1-0.5 range already tested
(final=0.000, best=0.490 - same collapse pattern) - checking whether that
holds set-wide too.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances with target_critic (Polyak tau=0.005) + replay buffer.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, target_critic Polyak tau=0.005 + replay buffer) ===")
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
                label_suffix="_balanced_targetcritic_polyak0.005_replaybuffer",
                target_critic_polyak_tau=0.005,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, target_critic Polyak tau=0.005) - failed: {failed} ===")
