"""
2026-09-09: tests whether the lr/lrc-specific TD-bootstrap collapse (see
chat, docs/SRL_Design.md) is driven by the length of the within-episode
bootstrap chain - an episode has many rolling-horizon iterations (steps),
far more than actual deadline-miss events, so the near-zero-reward TD
target (reward_mode="local") gets bootstrapped over a long chain before
ever touching a real reward. Doubling step_size (100 -> 200) and
batch_interval (200 -> 400) roughly halves the number of iterations per
episode for the same instance, shortening that chain - if the chain length
is the actual cause, this should reduce/remove the collapse on the
instances that collapsed at bi200/ss100 (lrc207, lr210, lrc108); lc108 is
included as the stable control (should stay stable either way).

Same gcn critic + MLP actor, gamma=0.99, tau=0.005, replay buffer
(capacity=40) as all other TD-bootstrap tests - only batch_interval/
step_size change.
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
    ACTOR_CHECKPOINT, TRAIN_INSTANCES, CRITIC_PRETRAIN_EPOCHS, EPISODES, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

BATCH_INTERVAL = 400
STEP_SIZE = 200

# lrc207, lr210, lrc108 collapsed to 0.0 at bi200/ss100 (seed 42); lc108 is
# the stable control from the same run.
TEST_INSTANCES = ["lrc207", "lr210", "lrc108", "lc108"]


def pretrain_shared_critic_bi400_ss200() -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "test_srl_td_bootstrap_bi400_ss200" / "critic_pretrain"

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
        print(f"[shared critic pretrain bi400/ss200] epoch {epoch} done")

    return critic


if __name__ == "__main__":
    print("Pretraining shared critic (bi400/ss200)...")
    shared_critic = pretrain_shared_critic_bi400_ss200()
    print("Shared critic pretraining done - running td_bootstrap (gamma=0.99, tau=0.005) at bi400/ss200 on the known-collapse instances + lc108 control.")

    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (td_bootstrap, gcn, bi400/ss200, gamma=0.99, tau=0.005) ===")
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
                label_suffix="_td_bootstrap_bi400_ss200_gamma0.99_tau0.005",
                critic_target_mode="td_bootstrap",
                gamma=0.99,
                target_critic_polyak_tau=0.005,
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing.")

    print("\n=== ALL bi400/ss200 TD-BOOTSTRAP TESTS DONE ===")
