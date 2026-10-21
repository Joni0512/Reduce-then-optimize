"""
One-off: reload twin_critic_epochs30_1's best (epoch 25) checkpoint and re-evaluate it on the
9 VAL_INSTANCES individually, to compare per-instance against RHO's own bi200/ss100 per-instance
rates (compute_rho_val_bi200.py) - the aggregate val=0.7085 doesn't say per-instance where the
actor gains/loses relative to RHO.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.eval_twin_critic_epochs30_per_instance
"""
import torch

from rtv_solver.pipeline.srl_training_loop import REPO_ROOT, _per_instance_service_rates
from rtv_solver.pipeline.srl_train_val_test_split import VAL_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42

output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / "twin_critic_epochs30_1"
checkpoint_path = output_dir / "best_actor_checkpoint.pt"

config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
active_feature_builder = select_feature_builder_class(config_template)
model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

rates = _per_instance_service_rates(VAL_INSTANCES, model, config_template, output_dir, epoch_num=25, tag="val_reeval")
for instance, rate in rates.items():
    print(f"  {instance}: actor_service_rate={rate:.4f}")
mean_rate = sum(rates.values()) / len(rates)
print(f"=== Actor (twin_critic_epochs30_1, best_epoch=25) mean over {len(rates)} VAL_INSTANCES = {mean_rate:.4f} ===")
