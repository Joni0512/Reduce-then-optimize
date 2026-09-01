"""
2026-08-31: replicates the target_critic (X=5) + replay-buffer combined
variant across 4 additional seeds (43-46), on top of the original seed=42
run (run_srl_balanced_targetcritic_replaybuffer_12instances.py) - see chat.
Same motivation as replicate_v1_top10_seeds.py: checks whether this
variant's results are robust or just a seed=42 artifact, given the known
run-to-run variance in this pipeline even at a fixed seed.

Runs the FULL balanced 12-instance test set per seed (this variant has only
one fixed hyperparameter config - target_critic_update_interval=5,
replay_capacity=40, critic_lr=1e-3 default, sigma=1.0 default - not 10
configs like the v1-sweep replication).
"""
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import csv

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

NEW_SEEDS = [43, 44, 45, 46]


def pretrain_shared_critic_seeded(seed: int) -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "replicate_targetcritic_replaybuffer_seeds" / f"seed{seed}" / "critic_pretrain"

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
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local")
        print(f"[seed {seed}] critic pretrain epoch {epoch} done")

    return critic


if __name__ == "__main__":
    results = []
    for seed in NEW_SEEDS:
        print(f"\n=== Pretraining shared critic for seed={seed} ===")
        shared_critic = pretrain_shared_critic_seeded(seed)

        for instance in TEST_INSTANCES:
            import copy
            critic = copy.deepcopy(shared_critic)
            label_suffix = f"_targetcritic_replaybuffer_seed{seed}"
            try:
                train(
                    instance=instance,
                    episodes=EPISODES,
                    batch_interval=BATCH_INTERVAL,
                    step_size=STEP_SIZE,
                    seed=seed,
                    actor_checkpoint=ACTOR_CHECKPOINT,
                    freeze_critic=False,
                    shared_critic=critic,
                    label_suffix=label_suffix,
                    target_critic_update_interval=5,
                    use_replay_buffer=True,
                    replay_capacity=40,
                    replay_batch_size=12,
                    replay_update_group_size=3,
                )
                label = f"{instance}_pretrained{label_suffix}"
                csv_path = REPO_ROOT / "outputs" / "train_srl_single_instance" / label / "srl_training_curves.csv"
                rows = list(csv.DictReader(open(csv_path)))
                final_sr = float(rows[-1]["service_rate"])
                best_sr = max(float(r["service_rate"]) for r in rows)
                print(f"seed={seed} instance={instance}: final={final_sr:.3f} best={best_sr:.3f}")
                results.append({"seed": seed, "instance": instance, "final_service_rate": final_sr, "best_service_rate": best_sr})
            except Exception as e:
                print(f"!!! seed={seed} instance={instance} FAILED: {e!r}")

    out_dir = REPO_ROOT / "outputs" / "replicate_targetcritic_replaybuffer_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "instance", "final_service_rate", "best_service_rate"])
        w.writeheader()
        w.writerows(results)
    print(f"\n=== DONE - saved {csv_path} ===")
