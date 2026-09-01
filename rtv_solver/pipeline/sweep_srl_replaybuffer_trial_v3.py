"""
2026-09-02: third wandb sweep for the SRL replay-buffer hyperparameters -
narrower, shifted-upward ranges based on v2's results (3sxp7i3o, n=49) - see
chat.

Changes from v2:
- critic_lr: 0.003 to 0.05 (was 0.001-0.03) - v2's top trials clustered in
  the upper half of its range (0.005-0.025), corr(log critic_lr, metric)
  stayed the strongest, most robust signal (+0.47) across the whole sweep -
  shifting the range up explores further past v2's ceiling instead of
  re-sampling the already-explored lower/middle region.
- replay_capacity: [60, 80, 100, 120] (was [20,30,40,60,80]) - v2's top-5
  leaned toward larger capacity (60/80/80/60/20 - mixed but skewed up) and
  its correlation (+0.27) was weak but consistently positive across every
  check this session - narrowing to the upper end + extending past v2's max
  (80) to see if the trend continues.
- sigma: unchanged (0.5-1.5) - v2 showed ~0 correlation, kept as a control/
  no-op dimension rather than fixing it, since a near-zero effect isn't the
  same as a proven null result at only n=49.

Runs on the FULL balanced 12-instance test set, same as v2 (not v1's
3-instance subset). User's call (2026-09-02): cluster-only, not local - see
submit_srl_replaybuffer_sweep_v3.sbatch.
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
            sigma=config.sigma,
            critic_lr=config.critic_lr,
            actor_checkpoint=ACTOR_CHECKPOINT,
            freeze_critic=False,
            shared_critic=critic,
            label_suffix=f"_sweepv3_{wandb.run.id}",
            use_replay_buffer=True,
            replay_capacity=config.replay_capacity,
            replay_batch_size=12,
            replay_update_group_size=3,
        )
        label = f"{instance}_pretrained_sweepv3_{wandb.run.id}"
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
