"""
2026-09-10: TD-bootstrap (gamma=0.99, tau=0.005) with GAT critic AND
reward_mode="local_positive" combined, on the full balanced 12-instance
test set, seed=43 (different from the local 4-instance diagnostic test,
which uses seed=42) - see chat and test_srl_td_bootstrap_gat_local_positive
.py's docstring for the motivation (checking whether combining GAT +
denser reward closes the remaining gcn+local_positive gap on lrc207/lr210,
or whether GAT alone already captures the full effect).
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

SEED = 43


def pretrain_shared_critic_gat_local_positive() -> torch.nn.Module:
    critic = CriticGNN(aggregator="gat")
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "run_srl_balanced_td_bootstrap_gat_local_positive_12instances_seed43" / "critic_pretrain"

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
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local_positive")
        print(f"[shared critic pretrain, gat, local_positive, seed43] epoch {epoch} done")

    return critic


if __name__ == "__main__":
    print("Pretraining shared GAT critic (reward_mode=local_positive, seed=43)...")
    shared_critic = pretrain_shared_critic_gat_local_positive()
    print("Shared GAT critic pretraining done - running all 12 test instances with td_bootstrap (gamma=0.99, tau=0.005, reward_mode=local_positive, seed=43).")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (balanced, gat, local_positive, td_bootstrap gamma=0.99 tau=0.005, seed=43) ===")
        try:
            critic = copy.deepcopy(shared_critic)
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                actor_checkpoint=ACTOR_CHECKPOINT,
                freeze_critic=False,
                shared_critic=critic,
                label_suffix="_balanced_td_bootstrap_gat_local_positive_gamma0.99_tau0.005_seed43",
                critic_target_mode="td_bootstrap",
                gamma=0.99,
                target_critic_polyak_tau=0.005,
                reward_mode="local_positive",
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing with remaining instances.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (balanced, gat, local_positive, td_bootstrap gamma=0.99 tau=0.005, seed=43) - failed: {failed} ===")
