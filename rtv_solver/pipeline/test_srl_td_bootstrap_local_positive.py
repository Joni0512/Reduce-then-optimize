"""
2026-09-10: TD-bootstrap (gamma=0.99, tau=0.005) with reward_mode=
"local_positive" instead of the default "local" - see chat, mc_return_
builder.py's build_local_positive() docstring, and docs/SRL_Design.md's TD
Bootstrap section. "local" gives r_t=0 almost everywhere (-1 only when a
request's deadline permanently passes unserved) - suspected as a root
cause of the observed lr/lrc-instance collapse, since the TD target then
becomes near-pure self-referential bootstrapping with rare real-reward
grounding. "local_positive" mirrors that with +1 only when a SERVICED
request's deadline passes - denser (most requests get served), tested
here on its own first (not combined with the -1 yet, per chat).

Same gcn critic + MLP actor, replay buffer (capacity=40) as all other
TD-bootstrap tests, on the same 4 diagnostic instances (lrc207, lr210,
lrc108 - collapsed under td_bootstrap+"local"; lc108 - stable control).
Critic pretrain also uses reward_mode="local_positive" (not the default
"local") - testing the reward end-to-end, so the critic's prior reflects
the new signal too, not just the SRL fine-tuning phase.
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
    ACTOR_CHECKPOINT, TRAIN_INSTANCES, CRITIC_PRETRAIN_EPOCHS, BATCH_INTERVAL, STEP_SIZE, EPISODES, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

TEST_INSTANCES = ["lrc207", "lr210", "lrc108", "lc108"]


def pretrain_shared_critic_local_positive() -> torch.nn.Module:
    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "test_srl_td_bootstrap_local_positive" / "critic_pretrain"

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
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local_positive")
        print(f"[shared critic pretrain, local_positive] epoch {epoch} done")

    return critic


if __name__ == "__main__":
    print("Pretraining shared critic (reward_mode=local_positive)...")
    shared_critic = pretrain_shared_critic_local_positive()
    print("Shared critic pretraining done - running td_bootstrap (gamma=0.99, tau=0.005, reward_mode=local_positive) on the diagnostic instances.")

    for instance in TEST_INSTANCES:
        print(f"\n=== {instance} (td_bootstrap, gcn, reward_mode=local_positive, gamma=0.99, tau=0.005) ===")
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
                label_suffix="_td_bootstrap_local_positive_gamma0.99_tau0.005",
                critic_target_mode="td_bootstrap",
                gamma=0.99,
                target_critic_polyak_tau=0.005,
                reward_mode="local_positive",
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing.")

    print("\n=== ALL local_positive + TD-BOOTSTRAP TESTS DONE ===")
