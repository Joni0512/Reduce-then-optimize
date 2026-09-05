"""
2026-09-05: fourth wandb sweep for the SRL hyperparameters - see chat.

Changes from v3:
- actor_lr: NEW sweep dimension, log_uniform 0.003-0.05 (same range as
  critic_lr, no prior data on actor_lr yet). Previously fixed at train()'s
  default (1e-4) and never swept in v1-v3.
- sigma: widened to log_uniform 0.01-10 (was uniform 0.5-1.5) - user's call,
  exploring the perturbation strength (used identically for both the
  perturb-and-MAP candidate sampling AND inside FenchelYoungLoss's own
  internal noise draw - confirmed in code there is only ONE sigma value,
  shared by both draws, not two separate tunable perturbation parameters).
- critic_lr: unchanged from v3 (log_uniform 0.003-0.05).
- replay_capacity: FIXED at 1000 (no longer swept) - user's call, despite
  the seed=42 12-instance result showing capacity=1000 underperforming
  capacity=40 (0.484 vs 0.580 final SR); a 4-seed replication of that
  finding is running in parallel on the cluster.

Runs on the FULL balanced 12-instance test set, same as v2/v3. Cluster-only,
same as v3 - see submit_srl_sweep_v4.sbatch.
"""
import copy

from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import csv
import torch
import wandb

from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.pretrain_and_save_shared_critic import OUTPUT_PATH as PRETRAINED_CRITIC_PATH
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import ACTOR_CHECKPOINT, TEST_INSTANCES, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED
from rtv_solver.pipeline.train_srl_single_instance import train, REPO_ROOT


def run_trial() -> None:
    wandb.init()
    config = wandb.config

    base_critic = CriticGNN()
    base_critic.load_state_dict(torch.load(PRETRAINED_CRITIC_PATH))

    final_service_rates = []
    for instance in TEST_INSTANCES:
        critic = copy.deepcopy(base_critic)
        train(
            instance=instance,
            episodes=EPISODES,
            batch_interval=BATCH_INTERVAL,
            step_size=STEP_SIZE,
            seed=SEED,
            actor_lr=config.actor_lr,
            sigma=config.sigma,
            critic_lr=config.critic_lr,
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=False,
            shared_critic=critic,
            label_suffix=f"_sweepv4_{wandb.run.id}",
            use_replay_buffer=True,
            replay_capacity=1000,
            replay_batch_size=12,
            replay_update_group_size=3,
        )
        label = f"{instance}_pretrained_sweepv4_{wandb.run.id}"
        csv_path = REPO_ROOT / "outputs" / "train_srl_single_instance" / label / "srl_training_curves.csv"
        rows = list(csv.DictReader(open(csv_path)))
        final_service_rate = float(rows[-1]["service_rate"])
        final_service_rates.append(final_service_rate)
        wandb.log({f"final_service_rate_{instance}": final_service_rate})

    mean_final_service_rate = sum(final_service_rates) / len(final_service_rates)
    wandb.log({"mean_final_service_rate": mean_final_service_rate})
    wandb.finish()


if __name__ == "__main__":
    run_trial()
