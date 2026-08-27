from rtv_solver.pipeline.train_srl_single_instance import train, REPO_ROOT

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

TEST_INSTANCES = [
    "lc107", "lc108", "lc109", "lr111", "lr112", "lrc107",
    "lc207", "lc208", "lr210", "lc206", "lr203", "lrc201",
]

EPISODES = 20
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42
TARGET_CRITIC_UPDATE_INTERVAL = 5

if __name__ == "__main__":
    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (target_critic, interval={TARGET_CRITIC_UPDATE_INTERVAL}) ===")
        try:
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                actor_checkpoint=ACTOR_CHECKPOINT,
                freeze_critic=False,
                target_critic_update_interval=TARGET_CRITIC_UPDATE_INTERVAL,
                label_suffix="_targetcritic_x5",
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (target_critic X=5) - failed: {failed} ===")
