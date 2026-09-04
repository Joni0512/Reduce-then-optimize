"""
2026-09-04: replay_capacity=1000 test on lrc207 - advisor's suggestion via
chat ("just set it to 1000, can't hurt"). At our instance sizes (~140-180
total buffered steps over a full 20-episode run), capacity=1000 means the
buffer essentially never fills up / never evicts anything - a de facto
unbounded buffer that keeps everything from episode 0 onward, unlike the
usual capacity=40 (which only remembers the last ~5 episodes' worth).
Otherwise identical setup to the standard Replay-Buffer-alone variant
(sigma=1.0, critic_lr=1e-3 defaults).
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print(f"Running {INSTANCE} with replay_capacity=1000 (de facto unbounded)...")

    train(
        instance=INSTANCE,
        episodes=EPISODES,
        batch_interval=BATCH_INTERVAL,
        step_size=STEP_SIZE,
        seed=SEED,
        actor_checkpoint=ACTOR_CHECKPOINT,
        freeze_critic=False,
        shared_critic=shared_critic,
        label_suffix="_balanced_replaybuffer_capacity1000",
        use_replay_buffer=True,
        replay_capacity=1000,
        replay_batch_size=12,
        replay_update_group_size=3,
    )

    print(f"\n=== {INSTANCE} (replay_capacity=1000) DONE ===")
