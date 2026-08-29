"""
2026-08-29: replay-buffer SRL sweep on the balanced 12-instance test set,
sigma=0.8 (instead of the usual 1.0) - see chat. On lrc207 alone, sigma=0.8
turned a flat collapse (final=0.020 at sigma=1.0) into an active recovery
within the 20 episodes (final=0.373) - testing whether that holds set-wide.
critic_lr stays at the default 1e-3 (that lever did NOT help on its own,
see chat).
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (sigma=0.8)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances with replay buffer, sigma=0.8.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, replay-buffer critic, sigma=0.8) ===")
        try:
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                sigma=0.8,
                actor_checkpoint=ACTOR_CHECKPOINT,
                freeze_critic=False,
                shared_critic=shared_critic,
                label_suffix="_balanced_replaybuffer_sigma0.8",
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, replay-buffer, sigma=0.8) - failed: {failed} ===")
