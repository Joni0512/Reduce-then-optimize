"""
2026-08-30: second, follow-up wandb sweep for the SRL replay-buffer
hyperparameters (see chat) - runs alongside the original
sweep_srl_replaybuffer_trial.py sweep (m8wzikdz), not replacing it.

Two changes from v1, based on 119 finished v1 trials:
- SWEEP_INSTANCES now the FULL balanced 12-instance test set (was a 3-instance
  subset in v1) - v1's sigma correlation drifted to ~0 with more trials and
  critic_lr stayed the only mildly robust signal, suggesting the 3-instance
  metric was too noisy; averaging over all 12 directly matches every other
  balanced-set result in this session, at the cost of each trial taking ~4x
  longer than v1's.
- critic_lr range shifted higher (1e-3 to 3e-2, was 1e-5 to 1e-2) - v1's top
  trials clustered somewhat above the old default (1e-3) far more often than
  below it; narrowing the search toward that region should sample it more
  densely instead of wasting trials on very low critic_lr, which consistently
  scored worse in v1.

sigma and replay_capacity ranges kept the same as v1 (weak/near-zero signal
either way, no reason yet to narrow them).
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

# 2026-08-30: user wants exactly the full balanced 12-instance test set here
# (same TEST_INSTANCES as run_srl_balanced_frozen_12instances.py etc.), not a
# hand-picked subset - see chat. Each trial is therefore much more expensive
# than v1's (12 instances x 20 episodes instead of 6), but the metric is
# directly comparable to every other balanced-set result in this session.
SWEEP_INSTANCES = TEST_INSTANCES


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
            label_suffix=f"_sweepv2_{wandb.run.id}",
            use_replay_buffer=True,
            replay_capacity=config.replay_capacity,
            replay_batch_size=12,
            replay_update_group_size=3,
        )
        label = f"{instance}_pretrained_sweepv2_{wandb.run.id}"
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
