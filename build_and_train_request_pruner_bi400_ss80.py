"""
2026-07-17: build the request-pruner dataset and run the hyperparameter
sweep for the new bi400/ss80 window config, reusing the exact same
pipeline build_window_specific_request_pruners.py uses for the other
configs - just pointed at bi400_ss80 paths instead of looping over the
fixed WINDOW_CONFIGS list (which stays untouched for the 3 existing configs).
"""
from pathlib import Path

from rtv_solver.pipeline.request_pruner_dataset_builder import build_dataset
from rtv_solver.pipeline.sweep_request_pruner import run_sweep

BATCH_INTERVAL, STEP_SIZE = 400, 80
label = f"bi{BATCH_INTERVAL}_ss{STEP_SIZE}"
rh_baseline_dir = Path(f"outputs/eval_rh_no_learning_{label}")
dataset_dir = Path(f"dataset/request_pruning_{label}")
model_output_dir = Path(f"outputs/request_pruner_mlp_{label}")

print(f"--- Building dataset from {rh_baseline_dir} -> {dataset_dir} ---")
build_dataset(rh_baseline_dir=rh_baseline_dir, output_dir=dataset_dir)

print(f"--- Sweeping hyperparameters -> {model_output_dir} ---")
run_sweep(dataset_dir=dataset_dir, output_dir=model_output_dir)

print("DONE")
