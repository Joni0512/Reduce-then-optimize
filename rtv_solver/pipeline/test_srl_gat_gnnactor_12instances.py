"""
2026-09-08: first test of SRL+GAT with a GNN actor instead of the usual MLP
actor - see chat. Same GAT critic setup as replicate_gat_nocliqueue_
12instances_seeds.py (aggregator="gat", no route-clique, baseline
hyperparameters: sigma=1.0, critic_lr=1e-3, actor_lr=1e-4,
replay_capacity=40, max_cardinality=2), only the actor changes: loads the
new GNN-actor SIL checkpoint (gcn, 2 layers) trained on the same balanced
mixed split (best val service rate 70.83%, vs. the MLP actor's 68.75% -
see chat), instead of ACTOR_CHECKPOINT (the MLP one). 1 seed (42), all 12
balanced test instances - first correctness/comparison check before
committing to a full multi-seed run.

2026-09-08: deliberately does NOT apply the FeatureBuilder.
ENABLE_PICKUP_SLACK_FEATURE=False monkey-patch that every other SRL/GAT
script in this pipeline uses (84 features) - the GNN actor checkpoint was
trained via plain `rtv_solver/main.py`, which uses FeatureBuilder's own
class default (ENABLE_PICKUP_SLACK_FEATURE=True, 85 features, see
feat_builder.py:110-123). Loading the checkpoint with the 84-feature patch
applied caused a state_dict shape mismatch (encoder.0.weight: [64,85] vs
[64,84]). Kept the pickup-slack feature ON (not retrained without it)
since the user's call was to keep this feature - note this on the results
slide: this GNN-actor run does NOT have identical features to the other
SRL/GAT baselines here (85 vs 84), so it isn't a perfectly isolated
actor-architecture-only comparison.
"""
import copy
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.run_srl_balanced_frozen_12instances import (
    TRAIN_INSTANCES, TEST_INSTANCES, CRITIC_PRETRAIN_EPOCHS, BATCH_INTERVAL, STEP_SIZE, EPISODES, SEED,
)
from rtv_solver.pipeline.train_srl_single_instance import train, MANIFEST_DIR, REPO_ROOT
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

# 2026-09-08: GNN-actor SIL checkpoint (gcn, 2 layers), mixed balanced split,
# best val service rate 70.83% at epoch 4 - see
# run_sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1.sh /
# submit_sil_training_mixed_balanced_gnn_gcn_l2_seed1.sbatch.
GNN_ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

MODEL_TYPE = "gnn"
GNN_AGGREGATOR = "gcn"
GNN_NUM_MESSAGE_PASSING_LAYERS = 2


def pretrain_shared_critic_gat() -> torch.nn.Module:
    critic = CriticGNN(aggregator="gat")
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    output_dir = REPO_ROOT / "outputs" / "test_srl_gat_gnnactor_12instances" / "critic_pretrain"

    for epoch in range(CRITIC_PRETRAIN_EPOCHS):
        for instance in TRAIN_INSTANCES:
            input_path = MANIFEST_DIR / f"{instance}.json"
            instance_output_dir = output_dir / instance
            instance_output_dir.mkdir(parents=True, exist_ok=True)
            config = Config(
                OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED,
                MODEL_TYPE=MODEL_TYPE, GNN_AGGREGATOR=GNN_AGGREGATOR, GNN_NUM_MESSAGE_PASSING_LAYERS=GNN_NUM_MESSAGE_PASSING_LAYERS,
            )
            setup_loggers(config.OUTPUT_DIR)
            set_seed(config.SEED, config.DEBUG)
            payload = PayloadParser.load_input_data(input_path)
            cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
            pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
            pipeline.load_model_weights(GNN_ACTOR_CHECKPOINT)
            pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=True, reward_mode="local")
        print(f"[gat+gnn-actor critic pretrain] epoch {epoch} done")

    return critic


if __name__ == "__main__":
    print("Pretraining GAT critic (with GNN actor) on the 12 balanced train instances...")
    shared_critic = pretrain_shared_critic_gat()
    print("Pretraining done - running all 12 test instances with aggregator=gat, GNN actor, no route-clique.")

    failed = []
    for instance in TEST_INSTANCES:
        print(f"=== {instance} (balanced, aggregator=gat, GNN actor, no route-clique) ===")
        try:
            critic = copy.deepcopy(shared_critic)
            train(
                instance=instance,
                episodes=EPISODES,
                batch_interval=BATCH_INTERVAL,
                step_size=STEP_SIZE,
                seed=SEED,
                actor_checkpoint=GNN_ACTOR_CHECKPOINT,
                freeze_critic=False,
                shared_critic=critic,
                label_suffix="_balanced_gat_gnnactor_seed42",
                use_replay_buffer=True,
                replay_capacity=40,
                replay_batch_size=12,
                replay_update_group_size=3,
                model_type=MODEL_TYPE,
                gnn_aggregator=GNN_AGGREGATOR,
                gnn_num_message_passing_layers=GNN_NUM_MESSAGE_PASSING_LAYERS,
            )
        except Exception as e:
            print(f"!!! {instance} FAILED: {e!r} - skipping, continuing.")
            failed.append(instance)

    print(f"\n=== ALL 12 INSTANCES DONE (gat, GNN actor, no route-clique) - failed: {failed} ===")
