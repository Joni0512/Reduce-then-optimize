"""
2026-08-31: target_critic (X=5) + replay-buffer combined, on the OLD
(unbalanced) train/test split - see chat. Mirrors
run_srl_unbalanced_pretrained_unfrozen_12instances.py's split (old mixed
actor checkpoint, 10x lc/2x lr/0x lrc train, 6x lc/4x lr/2x lrc test), but
adds target_critic (X=5, hard periodic copy) on top of the replay buffer -
same combination already tested on the NEW balanced split
(run_srl_balanced_targetcritic_replaybuffer_12instances.py: final=0.570,
best=0.734, no clear improvement over replay-buffer alone there). Checking
whether the combination behaves differently on the old, easier split.
"""
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

CRITIC_PRETRAIN_INSTANCES = ["lc101", "lc102", "lc103", "lc104", "lc105", "lc106", "lc201", "lc202", "lc203", "lc205", "lr201", "lr202"]
TEST_INSTANCES = [
    "lc107", "lc108", "lc109", "lr111", "lr112", "lrc107",   # class-1
    "lc207", "lc208", "lr210", "lc206", "lr203", "lrc201",   # class-2
]

CRITIC_PRETRAIN_EPOCHS = 10
EPISODES = 20
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42


def pretrain_shared_critic() -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "run_srl_unbalanced_targetcritic_replaybuffer_12instances" / "critic_pretrain"

    for epoch in range(CRITIC_PRETRAIN_EPOCHS):
        for instance in CRITIC_PRETRAIN_INSTANCES:
            input_path = MANIFEST_DIR / f"{instance}.json"
            instance_output_dir = output_dir / instance
            instance_output_dir.mkdir(parents=True, exist_ok=True)
            config = Config(OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
            setup_loggers(config.OUTPUT_DIR)
            set_seed(config.SEED, config.DEBUG)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
            pipeline.load_model_weights(ACTOR_CHECKPOINT)
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local")
        print(f"[shared critic pretrain] epoch {epoch} done")

    return critic


if __name__ == "__main__":
    print(f"Pretraining shared critic on {CRITIC_PRETRAIN_INSTANCES} ({CRITIC_PRETRAIN_EPOCHS} epochs)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - now running target_critic (X=5) + replay buffer for all test instances.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (unbalanced, target_critic X=5 + replay buffer) ===")
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
                label_suffix="_unbalanced_targetcritic_replaybuffer",
                target_critic_update_interval=5,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (unbalanced, target_critic + replay buffer) - failed: {failed} ===")
