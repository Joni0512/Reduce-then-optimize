"""
2026-09-21: Request Pruner for bi200/ss100 trained on CARDINALITY-3 RHO labels.

Same two steps as build_window_specific_request_pruners.py (dataset build, then
the full HP sweep with the winner retrained), but pointed at the cardinality-3
baseline logs (run_rh_baseline_data_bi200_ss100_c3.sh) and its own output
folders, so the existing cardinality-2 dataset/pruner are never touched:
    outputs/eval_rh_no_learning_bi200_ss100_c3/  -> dataset/request_pruning_bi200_ss100_c3/
    -> outputs/request_pruner_mlp_bi200_ss100_c3/

Run: python3 -m rtv_solver.pipeline.build_request_pruner_bi200_ss100_c3
"""
from pathlib import Path

from rtv_solver.pipeline.request_pruner_dataset_builder import build_dataset
from rtv_solver.pipeline.sweep_request_pruner import run_sweep

RH_BASELINE_DIR = Path("outputs/eval_rh_no_learning_bi200_ss100_c3")
DATASET_DIR = Path("dataset/request_pruning_bi200_ss100_c3")
MODEL_OUTPUT_DIR = Path("outputs/request_pruner_mlp_bi200_ss100_c3")


if __name__ == "__main__":
    print(f"--- Building dataset from {RH_BASELINE_DIR} -> {DATASET_DIR} ---")
    build_dataset(rh_baseline_dir=RH_BASELINE_DIR, output_dir=DATASET_DIR)
    print(f"\n--- Sweeping hyperparameters -> {MODEL_OUTPUT_DIR} ---")
    run_sweep(dataset_dir=DATASET_DIR, output_dir=MODEL_OUTPUT_DIR)
