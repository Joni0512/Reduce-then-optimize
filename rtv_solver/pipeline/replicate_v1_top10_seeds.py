"""
2026-08-31: replicate v1 sweep's top-10 hyperparameter configs across 4
additional seeds (43-46), on top of the original seed=42 - see chat.

Motivation: ALL 232 v1 sweep trials used the same hardcoded SEED=42 (both
for the critic pretrain AND the per-instance SRL runs - see
run_srl_balanced_frozen_12instances.py's module-level SEED constant, never
parameterized). The spread we saw across trials at "identical" hyperparameters
(e.g. lrc207: 0.373 vs 0.000 final in earlier tests) came from OTHER sources
of nondeterminism even at a fixed seed (documented in episode_buffer.py's
docstring - TripHandler's multiprocessing.Pool trip-cost generation, worker
completion order affects ILP tie-breaks even with the seed pinned). Varying
the seed here additionally exercises the actual seed-dependent randomness
(torch RNG for the perturb-and-MAP sigma noise, etc.) on top of that
existing noise floor - together, this checks whether the "best" v1 configs
are robustly good or just got lucky at seed=42.

CRITIC_PRETRAIN_EPOCHS/TRAIN_INSTANCES/ACTOR_CHECKPOINT reused directly from
run_srl_balanced_frozen_12instances.py. pretrain_shared_critic() there is
NOT reused as-is since it hardcodes SEED=42 - reimplemented here with a
seed parameter instead.
"""
import csv
from pathlib import Path

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
    ACTOR_CHECKPOINT, TRAIN_INSTANCES, CRITIC_PRETRAIN_EPOCHS, BATCH_INTERVAL, STEP_SIZE,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

SWEEP_INSTANCES = ["lc108", "lr210", "lrc108"]
NEW_SEEDS = [43, 44, 45, 46]

# top 10 configs from the v1 wandb sweep (m8wzikdz), by mean_final_service_rate
TOP10_CONFIGS = [
    {"run_id": "4pki8sph", "sigma": 0.777, "critic_lr": 0.00026, "replay_capacity": 40},
    {"run_id": "6zshxwsy", "sigma": 1.241, "critic_lr": 0.00481, "replay_capacity": 80},
    {"run_id": "q5ofghf5", "sigma": 0.589, "critic_lr": 0.00093, "replay_capacity": 80},
    {"run_id": "1ugj23v6", "sigma": 1.259, "critic_lr": 0.00657, "replay_capacity": 20},
    {"run_id": "52bpteoo", "sigma": 1.334, "critic_lr": 0.00939, "replay_capacity": 60},
    {"run_id": "veibkpe7", "sigma": 0.694, "critic_lr": 0.00017, "replay_capacity": 60},
    {"run_id": "rgdo4crm", "sigma": 1.037, "critic_lr": 0.00004, "replay_capacity": 40},
    {"run_id": "j7thdcm5", "sigma": 1.075, "critic_lr": 0.00020, "replay_capacity": 40},
    {"run_id": "sd0ldh50", "sigma": 0.624, "critic_lr": 0.00112, "replay_capacity": 60},
    {"run_id": "j6ule3f5", "sigma": 1.089, "critic_lr": 0.00005, "replay_capacity": 60},
]


def pretrain_shared_critic_seeded(seed: int) -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "replicate_v1_top10_seeds" / f"seed{seed}" / "critic_pretrain"

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

        for cfg in TOP10_CONFIGS:
            final_service_rates = []
            for instance in SWEEP_INSTANCES:
                import copy
                critic = copy.deepcopy(shared_critic)
                label_suffix = f"_replicate_seed{seed}_{cfg['run_id']}"
                try:
                    train(
                        instance=instance,
                        episodes=20,
                        batch_interval=BATCH_INTERVAL,
                        step_size=STEP_SIZE,
                        seed=seed,
                        sigma=cfg["sigma"],
                        critic_lr=cfg["critic_lr"],
                        actor_checkpoint=ACTOR_CHECKPOINT,
                        freeze_critic=False,
                        shared_critic=critic,
                        label_suffix=label_suffix,
                        use_replay_buffer=True,
                        replay_capacity=cfg["replay_capacity"],
                        replay_batch_size=12,
                        replay_update_group_size=3,
                    )
                    label = f"{instance}_pretrained{label_suffix}"
                    csv_path = REPO_ROOT / "outputs" / "train_srl_single_instance" / label / "srl_training_curves.csv"
                    rows = list(csv.DictReader(open(csv_path)))
                    final_service_rates.append(float(rows[-1]["service_rate"]))
                except Exception as e:
                    print(f"!!! seed={seed} config={cfg['run_id']} instance={instance} FAILED: {e!r}")

            if final_service_rates:
                mean_sr = sum(final_service_rates) / len(final_service_rates)
                print(f"seed={seed} config={cfg['run_id']}: mean_final_service_rate={mean_sr:.3f}")
                results.append({"seed": seed, "run_id": cfg["run_id"], "sigma": cfg["sigma"], "critic_lr": cfg["critic_lr"], "replay_capacity": cfg["replay_capacity"], "mean_final_service_rate": mean_sr})

    out_dir = REPO_ROOT / "outputs" / "replicate_v1_top10_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "run_id", "sigma", "critic_lr", "replay_capacity", "mean_final_service_rate"])
        w.writeheader()
        w.writerows(results)
    print(f"\n=== DONE - saved {csv_path} ===")
