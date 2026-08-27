"""
2026-08-25: frozen-critic SRL sweep over 12 held-out instances (6 class-1 +
6 class-2), using the mixed-class actor checkpoint (seed 1) - see chat.

The critic is pretrained ONCE (on the mixed 12-instance train split, same as
the mixed actor's own train set) and then reused, frozen, across all 12 test
instances - avoids repeating the expensive critic pretraining 12 times.
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

# same split used for all "mixed" runs this session
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
    output_dir = REPO_ROOT / "outputs" / "run_srl_frozen_critic_12instances" / "critic_pretrain"

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
    print("Shared critic pretraining done - now frozen for all test instances.")

    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (frozen shared critic) ===")
        train(
            instance=instance,
            episodes=EPISODES,
            batch_interval=BATCH_INTERVAL,
            step_size=STEP_SIZE,
            seed=SEED,
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=True,
            shared_critic=shared_critic,
            label_suffix="_clipped",
        )

    print("\n=== ALL 12 INSTANCES DONE ===")
