"""
2026-08-28: replay-buffer SRL sweep on the balanced 12-instance test set (see
chat) - full-set version of test_replay_buffer_lrc207.py, which showed the
buffer did NOT prevent lrc207's collapse. Runs the same setup
(capacity=40, batch=12, update_group_size=3, same pretrained shared critic
as run_srl_balanced_frozen_12instances.py) across all 12 balanced test
instances, to check whether that negative result holds set-wide or was
specific to lrc207.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as frozen/pretrained-unfrozen sweeps)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - each test instance gets its own fresh copy, trained via replay buffer.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, replay-buffer critic) ===")
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
                label_suffix="_balanced_replaybuffer",
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, replay-buffer) - failed: {failed} ===")
