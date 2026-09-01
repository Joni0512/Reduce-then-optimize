"""
2026-08-31: first test of Polyak (soft) target_critic update, tau=0.5 - see
chat. Deliberately simple/easy first value (half old target_critic, half new
live critic, blended every episode) instead of typical DDPG/TD3-style small
tau (~0.005) - just checking the mechanism works and has a visible effect
before tuning tau properly. Combined with the replay buffer (default HPs),
same setup as test_targetcritic_plus_replaybuffer_lrc207.py but with soft
blending instead of the hard X=5 copy tested there.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print(f"Running {INSTANCE} with target_critic (Polyak tau=0.5) + replay buffer combined...")

    train(
        instance=INSTANCE,
        episodes=EPISODES,
        batch_interval=BATCH_INTERVAL,
        step_size=STEP_SIZE,
        seed=SEED,
        actor_checkpoint=ACTOR_CHECKPOINT,
        freeze_critic=False,
        shared_critic=shared_critic,
        label_suffix="_balanced_targetcritic_polyak0.5_replaybuffer",
        target_critic_polyak_tau=0.5,
        use_replay_buffer=True,
        replay_capacity=40,
        replay_batch_size=12,
        replay_update_group_size=3,
    )

    print(f"\n=== {INSTANCE} (target_critic Polyak tau=0.5 + replay buffer) DONE ===")
