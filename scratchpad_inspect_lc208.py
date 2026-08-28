import torch
from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.srl_target_action import sample_candidate_assignments
from rtv_solver.pipeline.train_critic import MANIFEST_DIR
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ckpt = list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_class2_legacy_mlp_seed5").rglob("coaml_model_weights_best_val.pt"))[0]

train_instances = ["lc201", "lc202", "lc203", "lc205", "lr201", "lr202"]

critic = CriticGNN()
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
output_dir = REPO_ROOT / "outputs" / "inspect_lc208"
output_dir.mkdir(parents=True, exist_ok=True)

def run_episode(instance, train_critic):
    input_path = MANIFEST_DIR / f"{instance}.json"
    instance_output_dir = output_dir / instance
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(OUTPUT_DIR=instance_output_dir, MODE="coaml", BATCH_INTERVAL=200, STEP_SIZE=100, SEED=42)
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)
    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)
    pipeline = COAMLPipeline(config, cleared_payload, imitation_solution_path=input_path, critic=critic, critic_optimizer=critic_optimizer)
    pipeline.load_model_weights(str(ckpt))
    pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=train_critic, reward_mode="local")
    return pipeline

for epoch in range(10):
    for instance in train_instances:
        run_episode(instance, train_critic=True)
    print(f"pretrain epoch {epoch} done")

pipeline = run_episode("lc208", train_critic=False)

theta = pipeline.last_iteration_theta
oracle = pipeline.last_iteration_oracle
trip_costs = pipeline.last_iteration_trip_costs

print(f"theta shape: {theta.shape}, trip_costs: {len(trip_costs)}")
candidates = sample_candidate_assignments(theta, oracle, num_samples=10, sigma=1.0)

for i, y in enumerate(candidates):
    selected = (y[:len(trip_costs)] > 0.5).nonzero(as_tuple=True)[0].tolist()
    details = []
    for idx in selected:
        tc = trip_costs[idx]
        req_ids = tc.get_ordered_request_ids()
        details.append(f"veh{tc.vehicle_id}:{req_ids}")
    print(f"candidate {i}: {len(selected)} trips selected -> {details}")
