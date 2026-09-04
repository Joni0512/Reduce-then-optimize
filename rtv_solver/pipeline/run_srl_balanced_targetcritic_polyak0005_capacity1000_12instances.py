"""
2026-09-04: replay_capacity=1000 (de facto unbounded) COMBINED with
target_critic Polyak tau=0.005 (best tau found so far), full balanced
12-instance test set - see chat. New, previously-untested combination -
checking whether the two independent positive effects (larger buffer,
small-tau target_critic) add up.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances with replay_capacity=1000 + target_critic Polyak tau=0.005.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, replay_capacity=1000 + target_critic Polyak tau=0.005) ===")
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
                label_suffix="_balanced_targetcritic_polyak0.005_capacity1000",
                target_critic_polyak_tau=0.005,
                use_replay_buffer=True,
                replay_capacity=1000,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, target_critic Polyak tau=0.005 + capacity1000) - failed: {failed} ===")
