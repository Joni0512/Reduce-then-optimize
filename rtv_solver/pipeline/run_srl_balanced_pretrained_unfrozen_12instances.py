"""
2026-08-28: "pretrained but still co-adapting" SRL sweep (see chat) - fills
the missing quadrant of our 2x2 design (pretrained-start x co-adapting).

Same shared critic pretraining as run_srl_balanced_frozen_12instances.py
(reused directly from there), but freeze_critic=False here: each test
instance gets a FRESH deepcopy of the pretrained critic (via
train_srl_single_instance.py's shared_critic deepcopy, added 2026-08-28) and
that copy keeps training (train_critic=True) during that instance's 20 SRL
episodes - isolates whether moving-target instability persists even when the
critic starts from a competent, pretrained point, instead of from scratch.
"""
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train

if __name__ == "__main__":
    print("Pretraining shared critic (same as frozen sweep)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - each test instance gets its own fresh copy, then keeps co-adapting.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, pretrained-but-unfrozen critic) ===")
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
                label_suffix="_balanced_pretrained_unfrozen",
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, pretrained-unfrozen) - failed: {failed} ===")
