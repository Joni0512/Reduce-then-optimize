"""
2026-09-02: replicates the v2 wandb sweep's (3sxp7i3o) top-5 hyperparameter
configs across 4 additional seeds (43-46), on top of the original seed=42 -
see chat. Same motivation as replicate_v1_top10_seeds.py: checks whether
these configs are robustly good or just seed=42 artifacts, given the known
run-to-run variance in this pipeline even at a fixed seed.

Unlike v1's replication (only 3 instances per trial), this runs the FULL
balanced 12-instance test set per config per seed - v2 already sweeps on all
12 instances, so this keeps the metric directly comparable.

User's call (2026-09-01): run this on the CLUSTER, not locally - see
submit_replicate_v2_top5_seeds.sbatch.
"""
import copy
import csv

from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    pretrain_shared_critic, ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE,
)
from rtv_solver.pipeline.train_srl_single_instance import train, REPO_ROOT

NEW_SEEDS = [43, 44, 45, 46]

# top 5 configs from the v2 wandb sweep (3sxp7i3o), by mean_final_service_rate
TOP5_CONFIGS = [
    {"run_id": "ld0vlsri", "sigma": 0.5788, "critic_lr": 0.02519, "replay_capacity": 60},
    {"run_id": "6sdp10wj", "sigma": 1.3143, "critic_lr": 0.00732, "replay_capacity": 80},
    {"run_id": "4bbcdcty", "sigma": 1.2367, "critic_lr": 0.00942, "replay_capacity": 20},
    {"run_id": "pzmr4v3p", "sigma": 0.6873, "critic_lr": 0.02333, "replay_capacity": 60},
    {"run_id": "fzltz6y8", "sigma": 0.8311, "critic_lr": 0.00538, "replay_capacity": 80},
]

if __name__ == "__main__":
    results = []
    for seed in NEW_SEEDS:
        print(f"\n=== Pretraining shared critic for seed={seed} ===")
        # NOTE: pretrain_shared_critic() hardcodes SEED=42 internally (module-level
        # constant in run_srl_balanced_frozen_12instances.py) - same critic
        # pretrain reused across seeds here, only the per-instance SRL fine-tuning
        # below actually varies by seed. (Matches replicate_targetcritic_replaybuffer_seeds.py's
        # seeded variant if more precise per-seed pretraining is wanted instead -
        # kept simple/consistent with v2 sweep's own trial script here, which
        # also always uses the fixed pretrained critic loaded from disk.)
        shared_critic = pretrain_shared_critic()

        for cfg in TOP5_CONFIGS:
            final_service_rates = []
            for instance in TEST_INSTANCES:
                critic = copy.deepcopy(shared_critic)
                label_suffix = f"_v2replicate_seed{seed}_{cfg['run_id']}"
                try:
                    train(
                        instance=instance,
                        episodes=EPISODES,
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
                print(f"seed={seed} config={cfg['run_id']}: mean_final_service_rate={mean_sr:.3f} (n_instances={len(final_service_rates)})")
                results.append({"seed": seed, "run_id": cfg["run_id"], "sigma": cfg["sigma"], "critic_lr": cfg["critic_lr"], "replay_capacity": cfg["replay_capacity"], "mean_final_service_rate": mean_sr})

    out_dir = REPO_ROOT / "outputs" / "replicate_v2_top5_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "run_id", "sigma", "critic_lr", "replay_capacity", "mean_final_service_rate"])
        w.writeheader()
        w.writerows(results)
    print(f"\n=== DONE - saved {csv_path} ===")
