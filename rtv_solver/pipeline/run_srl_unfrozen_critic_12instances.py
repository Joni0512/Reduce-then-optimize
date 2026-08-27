"""
2026-08-25: comparison run to run_srl_frozen_critic_12instances.py - SAME 12
held-out instances, same mixed-class actor checkpoint (seed 1), same
episodes/hyperparameters, but WITHOUT freezing the critic (critic co-adapts
with the actor every episode, the original moving-target setup) - see chat.

Unlike the frozen-critic sweep, each instance gets its OWN freshly-built
critic here (train_srl_single_instance.py's default behavior) since the
whole point is to let actor and critic co-adapt per instance, not share a
fixed critic across instances.
"""
from rtv_solver.pipeline.train_srl_single_instance import train, REPO_ROOT

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

TEST_INSTANCES = [
    "lc107", "lc108", "lc109", "lr111", "lr112", "lrc107",   # class-1
    "lc207", "lc208", "lr210", "lc206", "lr203", "lrc201",   # class-2
]

EPISODES = 20
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42

if __name__ == "__main__":
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (unfrozen, co-adapting critic) ===")
        train(
            instance=instance,
            episodes=EPISODES,
            batch_interval=BATCH_INTERVAL,
            step_size=STEP_SIZE,
            seed=SEED,
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=False,
            label_suffix="_mixed_clipped",
        )

    print("\n=== ALL 12 INSTANCES DONE (unfrozen) ===")
