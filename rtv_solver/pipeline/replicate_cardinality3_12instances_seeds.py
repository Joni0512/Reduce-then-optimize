"""
2026-09-07: tests max_cardinality=3 (instead of the Config default of 2,
used unchanged by every SRL run this session so far) - see chat, motivated
by Table 4 (RHO/SIL/OPT comparison paper) showing RHO's service rate rising
noticeably from cardinality 2 to 3 to 4 at horizon=200/step=100. Plain gcn
aggregator (CriticGNN default), baseline hyperparameters otherwise
(sigma=1.0, critic_lr=1e-3, actor_lr=1e-4, replay_capacity=40 - all train()
defaults, unchanged) - companion script
replicate_gat_nocliqueue_cardinality3_seeds.py tests the same cardinality=3
change WITH aggregator="gat" for comparison. 5 seeds, all 12 balanced test
instances.
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import copy
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    ACTOR_CHECKPOINT, TRAIN_INSTANCES, TEST_INSTANCES, CRITIC_PRETRAIN_EPOCHS, BATCH_INTERVAL, STEP_SIZE, EPISODES,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

SEEDS = [42, 43, 44, 45, 46]
MAX_CARDINALITY = 3


def pretrain_shared_critic_seeded(seed: int) -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "replicate_cardinality3_12instances_seeds" / f"seed{seed}" / "critic_pretrain"

    for epoch in range(CRITIC_PRETRAIN_EPOCHS):
        for instance in TRAIN_INSTANCES:
            input_path = MANIFEST_DIR / f"{instance}.json"
            instance_output_dir = output_dir / instance
            instance_output_dir.mkdir(parents=True, exist_ok=True)
            config = Config(
                OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=seed,
                MAX_CARDINALITY=MAX_CARDINALITY,
            )
            setup_loggers(config.OUTPUT_DIR)
            set_seed(config.SEED, config.DEBUG)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
            pipeline.load_model_weights(ACTOR_CHECKPOINT)
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local")
        print(f"[seed {seed}] critic pretrain epoch {epoch} done")

    return critic


if __name__ == "__main__":
    for seed in SEEDS:
        print(f"\n=== Pretraining shared critic (cardinality={MAX_CARDINALITY}) for seed={seed} ===")
        shared_critic = pretrain_shared_critic_seeded(seed)

        for instance in TEST_INSTANCES:
            print(f"=== seed={seed} instance={instance} (balanced, gcn, cardinality={MAX_CARDINALITY}) ===")
            try:
                critic = copy.deepcopy(shared_critic)
                train(
                    instance=instance,
                    episodes=EPISODES,
                    batch_interval=BATCH_INTERVAL,
                    step_size=STEP_SIZE,
                    seed=seed,
                    actor_checkpoint=ACTOR_CHECKPOINT,
                    freeze_critic=False,
                    shared_critic=critic,
                    label_suffix=f"_balanced_cardinality3_seed{seed}",
                    use_replay_buffer=True,
                    replay_capacity=40,
                    replay_batch_size=12,
                    replay_update_group_size=3,
                    max_cardinality=MAX_CARDINALITY,
                )
            except Exception as e:
                print(f"!!! seed={seed} instance={instance} FAILED: {e!r} - skipping, continuing.")

    print("\n=== ALL SEEDS x 12 INSTANCES DONE (gcn, cardinality=3) ===")
