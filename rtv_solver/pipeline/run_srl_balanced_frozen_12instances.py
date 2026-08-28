"""
2026-08-28: frozen-critic SRL sweep on the new BALANCED train/test split
(2 lc + 2 lr + 2 lrc per class, both sets - see chat, fixes the earlier
lc-heavy/lrc-absent split). Critic pretrained once on the 12 balanced train
instances, then frozen for all 12 balanced test instances.
"""
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

TRAIN_INSTANCES = ["lc101", "lc102", "lr101", "lr102", "lrc101", "lrc102", "lc201", "lc202", "lr201", "lr202", "lrc202", "lrc203"]
TEST_INSTANCES = ["lc107", "lc108", "lr111", "lr112", "lrc107", "lrc108", "lc207", "lc208", "lr203", "lr210", "lrc201", "lrc207"]

CRITIC_PRETRAIN_EPOCHS = 10
EPISODES = 20
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42


def pretrain_shared_critic() -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "run_srl_balanced_frozen_12instances" / "critic_pretrain"

    for epoch in range(CRITIC_PRETRAIN_EPOCHS):
        for instance in TRAIN_INSTANCES:
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
    print(f"Pretraining shared critic on {TRAIN_INSTANCES} ({CRITIC_PRETRAIN_EPOCHS} epochs)...")
    shared_critic = pretrain_shared_critic()
    print("Shared critic pretraining done - now frozen for all test instances.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, frozen shared critic) ===")
        try:
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                actor_checkpoint=ACTOR_CHECKPOINT,
                freeze_critic=True,
                shared_critic=shared_critic,
                label_suffix="_balanced_frozen",
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, frozen) - failed: {failed} ===")
