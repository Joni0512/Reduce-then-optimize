"""
2026-08-29: one wandb-sweep trial for the SRL replay-buffer hyperparameters
(replay_capacity, critic_lr, sigma) - see chat. Run via `wandb agent
<sweep-id>` (see sweep_srl_replaybuffer.yaml at repo root), any number of
times, on any machine (local, LRZ cluster partitions serial/serial_long) -
each invocation asks the wandb sweep controller for the next hyperparameter
combination and reports back a single scalar metric.

Uses a SMALL, fixed, mixed instance set (SWEEP_INSTANCES below - one stable
lc instance + two fragile lr/lrc instances) instead of just lrc207 alone:
tuning against lrc207's own SRL fine-tuning result would mean picking
hyperparameters specifically for the one instance being "tested" on (see
chat) - not a generalizable signal.

Loads the ALREADY-pretrained shared critic from
pretrain_and_save_shared_critic.py instead of re-pretraining per trial - that
phase is expensive (10 epochs x 12 instances) and its own critic_lr (fixed at
1e-3 inside pretrain_shared_critic()) is unrelated to the swept critic_lr,
which only affects the per-instance SRL fine-tuning phase below.

2026-08-29: FEATURE_SIZE monkey-patch (see pretrain_and_save_shared_critic.py
docstring) - commit 69b521f changed feat_builder.py's default
ENABLE_PICKUP_SLACK_FEATURE to True after ACTOR_CHECKPOINT was trained,
breaking checkpoint loading. Patched back to False here so this sweep stays
comparable to last night's SRL results, pending the user's separate decision
on whether to retrain the actor checkpoint with the new feature enabled.
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
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import ACTOR_CHECKPOINT, EPISODES, BATCH_INTERVAL, STEP_SIZE, SEED
from rtv_solver.pipeline.train_srl_single_instance import train, REPO_ROOT

SWEEP_INSTANCES = ["lc108", "lr210", "lrc108"]  # 1 stable + 2 fragile, see docstring


def run_trial() -> None:
    wandb.init()
    config = wandb.config

    base_critic = CriticGNN()
    base_critic.load_state_dict(torch.load(PRETRAINED_CRITIC_PATH))

    final_service_rates = []
    for instance in SWEEP_INSTANCES:
        critic = copy.deepcopy(base_critic)
        train(
            instance=instance,
            episodes=EPISODES,
            batch_interval=BATCH_INTERVAL,
            step_size=STEP_SIZE,
            seed=SEED,
            sigma=config.sigma,
            critic_lr=config.critic_lr,
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=False,
            shared_critic=critic,
            label_suffix=f"_sweep_{wandb.run.id}",
            use_replay_buffer=True,
            replay_capacity=config.replay_capacity,
            replay_batch_size=12,
            replay_update_group_size=3,
        )
        label = f"{instance}_pretrained_sweep_{wandb.run.id}"
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
