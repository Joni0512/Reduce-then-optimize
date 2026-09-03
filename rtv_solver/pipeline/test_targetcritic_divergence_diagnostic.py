"""
2026-09-03: diagnostic test - does target_critic (Polyak tau=0.3) drift away
from the live critic over episodes, and does that drift track with the
service-rate collapse? See chat.

Logs critic_tc_corr (Pearson correlation between critic's and target_critic's
predictions on the same episode steps) and critic_tc_mean_abs_diff per
episode, alongside service_rate, in srl_training_curves.csv.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print(f"Running {INSTANCE} with target_critic (Polyak tau=0.3) + replay buffer, divergence diagnostic active...")

    train(
        instance=INSTANCE,
        episodes=EPISODES,
        batch_interval=BATCH_INTERVAL,
        step_size=STEP_SIZE,
        seed=SEED,
        actor_checkpoint=ACTOR_CHECKPOINT,
        freeze_critic=False,
        shared_critic=shared_critic,
        label_suffix="_balanced_targetcritic_polyak0.3_diagnostic",
        target_critic_polyak_tau=0.3,
        use_replay_buffer=True,
        replay_capacity=40,
        replay_batch_size=12,
        replay_update_group_size=3,
    )

    print(f"\n=== {INSTANCE} (target_critic Polyak tau=0.3, divergence diagnostic) DONE ===")
