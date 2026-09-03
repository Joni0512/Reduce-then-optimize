"""
2026-09-04: target_critic (Polyak tau=0.3) + replay buffer, on the full
balanced 12-instance test set, WITH the critic/target_critic divergence
diagnostic active - see chat. Extends the single-instance (lrc207, 5 seeds)
divergence finding to the full test set: does the "strongest disagreement
happens early, both networks converge by the end regardless of outcome"
pattern hold across instances, not just lrc207?

The critic_tc_corr/critic_tc_mean_abs_diff columns are logged automatically
by coaml_pipeline.py/train_srl_single_instance.py whenever target_critic is
a distinct object and the replay buffer is active - no extra wiring needed
here.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as other balanced sweeps)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - running all 12 test instances with target_critic (Polyak tau=0.3) + replay buffer, divergence diagnostic active.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, target_critic Polyak tau=0.3, divergence diagnostic) ===")
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
                label_suffix="_balanced_targetcritic_polyak0.3_diagnostic_all12",
                target_critic_polyak_tau=0.3,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, target_critic Polyak tau=0.3, divergence diagnostic) - failed: {failed} ===")
