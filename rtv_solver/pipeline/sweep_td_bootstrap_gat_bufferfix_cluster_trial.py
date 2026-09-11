"""
2026-09-11: pure-training half of the GAT+TD-bootstrap+bufferfix wandb
sweep, meant to run as a normal sbatch job on an LRZ compute node - see
chat. Compute nodes have no internet access, so this script does NOT talk
to wandb at all (unlike sweep_td_bootstrap_gat_bufferfix_trial.py, which
runs the whole thing locally on the Mac, wandb.init()/log() included).

Takes gamma/tau/a run_id via argv, does the exact same pretrain + 12-
instance training as the local trial function, and writes ONE JSON file
with the resulting mean_final_service_rate plus per-instance rates - a
coordinator process running on the LOGIN node (which DOES have internet)
reads that file back and reports it to the SAME wandb sweep the Mac agent
pulls from, so local and cluster share the sweep's trial budget - see
sweep_td_bootstrap_gat_bufferfix_coordinator.py.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.sweep_td_bootstrap_gat_bufferfix_cluster_trial <run_id> <gamma> <tau>
"""
import copy
import csv
import json
import sys

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
    ACTOR_CHECKPOINT, TRAIN_INSTANCES, TEST_INSTANCES, CRITIC_PRETRAIN_EPOCHS, BATCH_INTERVAL, STEP_SIZE, EPISODES, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers


def pretrain_shared_critic_gat(run_id: str) -> torch.nn.Module:
    critic = CriticGNN(aggregator="gat")
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "sweep_td_bootstrap_gat_bufferfix_cluster" / run_id / "critic_pretrain"

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
        print(f"[cluster trial {run_id}] critic pretrain epoch {epoch} done")

    return critic


def main() -> None:
    run_id = sys.argv[1]
    gamma = float(sys.argv[2])
    tau = float(sys.argv[3])
    print(f"=== cluster trial {run_id}: gamma={gamma} tau={tau} ===")

    base_critic = pretrain_shared_critic_gat(run_id)

    final_service_rates = {}
    for instance in TEST_INSTANCES:
        critic = copy.deepcopy(base_critic)
        train(
            instance=instance,
            episodes=EPISODES,
            batch_interval=BATCH_INTERVAL,
            step_size=STEP_SIZE,
            seed=SEED,
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=False,
            shared_critic=critic,
            label_suffix=f"_gat_bufferfix_sweep_cluster_{run_id}",
            critic_target_mode="td_bootstrap",
            gamma=gamma,
            target_critic_polyak_tau=tau,
            use_replay_buffer=True,
            replay_capacity=40,
            replay_batch_size=12,
            replay_update_group_size=3,
        )
        label = f"{instance}_pretrained_gat_bufferfix_sweep_cluster_{run_id}"
        csv_path = REPO_ROOT / "outputs" / "train_srl_single_instance" / label / "srl_training_curves.csv"
        rows = list(csv.DictReader(open(csv_path)))
        final_service_rates[instance] = float(rows[-1]["service_rate"])
        print(f"=== cluster trial {run_id}: {instance} done, service_rate={final_service_rates[instance]:.3f} ===")

    mean_final_service_rate = sum(final_service_rates.values()) / len(final_service_rates)

    result_dir = REPO_ROOT / "outputs" / "sweep_td_bootstrap_gat_bufferfix_cluster" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump({
            "run_id": run_id, "gamma": gamma, "tau": tau,
            "final_service_rates": final_service_rates,
            "mean_final_service_rate": mean_final_service_rate,
        }, f, indent=2)
    print(f"=== cluster trial {run_id} DONE: mean_final_service_rate={mean_final_service_rate:.4f} - wrote {result_path} ===")


if __name__ == "__main__":
    main()
