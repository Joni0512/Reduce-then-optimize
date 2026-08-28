"""
2026-08-27: NYC request-pruner hyperparameter sweep (Phase 1, step 3b of the
NYC RequestPruner readiness plan). Same pattern as
build_and_train_request_pruner_bi200_ss40.py, pointed at the NYC dataset
(dataset/request_pruning_nyc/, 4min/8min config, geographic=True features)
instead of building a new one - see build_nyc_request_pruner_dataset.py for
how that dataset was built.
"""
from pathlib import Path

from rtv_solver.pipeline.sweep_request_pruner import run_sweep

dataset_dir = Path("dataset/request_pruning_nyc")
output_dir = Path("outputs/request_pruner_mlp_nyc")

print(f"--- Sweeping hyperparameters: {dataset_dir} -> {output_dir} ---")
run_sweep(dataset_dir=dataset_dir, output_dir=output_dir)
print("DONE")
