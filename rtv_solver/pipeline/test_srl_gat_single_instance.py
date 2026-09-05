"""
2026-09-05: first correctness/comparison test for the new GATLayer
(aggregator="gat" in CriticGNN) - see chat and docs/SRL_Design.md's GAT plan
section. Runs on ONE instance (lrc207, our recurring test case this
session) with the established Replay-Buffer-alone baseline hyperparameters
(sigma=1.0, critic_lr=1e-3, actor_lr=1e-4, replay_capacity=40 - all train()
defaults, unchanged), so the only difference vs. the known gcn baseline is
the aggregator. Deliberately WITHOUT the route-clique edge extension yet
(still the existing chain-only Request-Request edges) - user's call, test
the attention mechanism itself first before also changing the neighborhood.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    ACTOR_CHECKPOINT, TRAIN_INSTANCES, CRITIC_PRETRAIN_EPOCHS, BATCH_INTERVAL, STEP_SIZE, EPISODES, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

TEST_INSTANCE = "lrc207"


def pretrain_shared_critic_gat() -> torch.nn.Module:
    critic = CriticGNN(aggregator="gat")
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "test_srl_gat_single_instance" / "critic_pretrain"

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
        print(f"[gat critic pretrain] epoch {epoch} done")

    return critic


if __name__ == "__main__":
    print("Pretraining GAT critic on the 12 balanced train instances...")
    gat_critic = pretrain_shared_critic_gat()
    print(f"Pretraining done - running {TEST_INSTANCE} with aggregator=gat (baseline HPs, no route-clique yet).")

    train(
        instance=TEST_INSTANCE,
        episodes=EPISODES,
        batch_interval=BATCH_INTERVAL,
        step_size=STEP_SIZE,
        seed=SEED,
        actor_checkpoint=ACTOR_CHECKPOINT,
        freeze_critic=False,
        shared_critic=gat_critic,
        label_suffix="_gat_test_nocilque",
        use_replay_buffer=True,
        replay_capacity=40,
        replay_batch_size=12,
        replay_update_group_size=3,
    )
    print(f"\n=== {TEST_INSTANCE} (aggregator=gat, no route-clique) DONE ===")
