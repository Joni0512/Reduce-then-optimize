"""
2026-09-11: GAT critic + TD-bootstrap (gamma=0.99, tau=0.005) +
reward_mode="local_positive", AFTER the replay-buffer staleness fix, for
ONE seed passed via argv - see chat. Companion to
run_srl_balanced_td_bootstrap_gat_local_positive_nobufferfix_single_seed.py
(same config, but with use_stale_td_target=True to reproduce the pre-fix
bug) - together they give a clean bufferfix vs. no-bufferfix comparison
under local_positive, mirroring the earlier gat_bufferfix_12instances vs.
gat_12instances comparison done under reward_mode="local".

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.run_srl_balanced_td_bootstrap_gat_local_positive_bufferfix_single_seed <seed>
"""
import sys

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


def pretrain_shared_critic_gat_seeded(seed: int) -> torch.nn.Module:
    critic = CriticGNN(aggregator="gat")
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "run_srl_balanced_td_bootstrap_gat_local_positive_bufferfix_12instances" / f"seed{seed}" / "critic_pretrain"

    for epoch in range(CRITIC_PRETRAIN_EPOCHS):
        for instance in TRAIN_INSTANCES:
            input_path = MANIFEST_DIR / f"{instance}.json"
            instance_output_dir = output_dir / instance
            instance_output_dir.mkdir(parents=True, exist_ok=True)
            config = Config(OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=seed)
            setup_loggers(config.OUTPUT_DIR)
            set_seed(config.SEED, config.DEBUG)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
            pipeline.load_model_weights(ACTOR_CHECKPOINT)
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local_positive")
        print(f"[seed {seed}] critic pretrain epoch {epoch} done")

    return critic


if __name__ == "__main__":
    seed = int(sys.argv[1])
    print(f"\n=== Pretraining shared GAT critic for seed={seed} (local_positive, bufferfix) ===")
    shared_critic = pretrain_shared_critic_gat_seeded(seed)

    for instance in TEST_INSTANCES:
        print(f"=== seed={seed} instance={instance} (balanced, gat, local_positive, td_bootstrap gamma=0.99 tau=0.005, bufferfix) ===")
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
                label_suffix=f"_balanced_td_bootstrap_gat_local_positive_bufferfix_gamma0.99_tau0.005_seed{seed}",
                critic_target_mode="td_bootstrap",
                gamma=0.99,
                target_critic_polyak_tau=0.005,
                reward_mode="local_positive",
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
                use_stale_td_target=False,
            )
        except Exception as e:
            print(f"!!! seed={seed} instance={instance} FAILED: {e!r} - skipping, continuing.")

    print(f"\n=== SEED {seed} x 12 INSTANCES (gat, local_positive, bufferfix) DONE ===")
