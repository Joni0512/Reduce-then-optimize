"""
2026-08-28: replay-buffer test on lrc207 (see chat) - this instance collapsed
hardest of all 12 balanced-split test instances, both frozen (0.510 -> 0.000
by episode 5) and pretrained-unfrozen (0.451 -> 0.157). Tests whether a
cross-episode Monte Carlo replay buffer (capacity=40, batch=12,
update_group_size=3 - see replay_buffer.py) stops that collapse, using the
SAME pretrained shared critic as run_srl_balanced_frozen_12instances.py so
the comparison is apples-to-apples.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

INSTANCE = "lrc207"

if __name__ == "__main__":
    print("Pretraining shared critic (same as balanced sweep)...")
    shared_critic = pretrain_shared_critic()
    print(f"Shared critic pretraining done - running {INSTANCE} with replay-buffer critic training.")

    train(
        instance=INSTANCE,
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

    print(f"\n=== {INSTANCE} (balanced, replay-buffer critic) DONE ===")
