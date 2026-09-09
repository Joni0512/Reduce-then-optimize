"""
2026-09-09: wandb sweep for TD-bootstrap critic target (critic_target_mode
="td_bootstrap") - see chat. Sweeps gamma (0.95-0.99) and the Polyak
target_critic tau (0.001-0.005), at bi200/ss100 (the default rolling-
horizon config all TD-bootstrap tests so far used) - checking whether some
combination in this range avoids the lr/lrc-instance collapse seen at
gamma=0.99/tau=0.005 (see docs/SRL_Design.md's TD Bootstrap plan section).

Reuses the same pretrained shared critic as the other SRL sweeps
(pretrain_and_save_shared_critic.py's output) - that critic was pretrained
at bi200/ss100 too, so no re-pretrain needed per trial. replay_capacity
fixed at 40 (matching all other TD-bootstrap tests, not the 1000 used by
the non-td_bootstrap replay-buffer sweeps).

Runs on the full balanced 12-instance test set, same as v3/v4. Cluster-only
- see submit_sweep_td_bootstrap.sbatch.
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
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=False,
            shared_critic=critic,
            label_suffix=f"_tdbootstrap_sweep_{wandb.run.id}",
            critic_target_mode="td_bootstrap",
            gamma=config.gamma,
            target_critic_polyak_tau=config.tau,
            use_replay_buffer=True,
            replay_capacity=40,
            replay_batch_size=12,
            replay_update_group_size=3,
        )
        label = f"{instance}_pretrained_tdbootstrap_sweep_{wandb.run.id}"
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
