"""
2026-09-04: target_critic Polyak with a much smaller, classic DDPG/TD3-style
tau=0.005 (all earlier tests covered 0.1-0.5) - see chat. Checks whether
very slow blending behaves differently from the 0.1-0.5 range, which
performed uniformly poorly regardless of value (arguing against a simple
"too much staleness" explanation - see the weight-averaging hypothesis
discussed in the meeting deck).
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print(f"Running {INSTANCE} with target_critic (Polyak tau=0.005) + replay buffer...")

    train(
        instance=INSTANCE,
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

    print(f"\n=== {INSTANCE} (target_critic Polyak tau=0.005) DONE ===")
