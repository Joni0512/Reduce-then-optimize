"""
2026-08-31: target_critic (X=5, Option B, hard periodic copy) combined with
the replay buffer, on the full balanced 12-instance test set - see chat.
First-ever combination of the two mechanisms at scale (single-instance test
on lrc207 showed no final-episode improvement but a notably better best-ep
peak, 0.627 vs 0.510 frozen/0.490 replay-buffer-alone - checking whether
that holds set-wide).

Both mechanisms already independent, unguarded params in
train_srl_single_instance.train() - no code change needed, just passing both.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances with target_critic (X=5) + replay buffer.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, target_critic X=5 + replay buffer) ===")
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
                label_suffix="_balanced_targetcritic_x5_replaybuffer",
                target_critic_update_interval=5,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, target_critic X=5 + replay buffer) - failed: {failed} ===")
