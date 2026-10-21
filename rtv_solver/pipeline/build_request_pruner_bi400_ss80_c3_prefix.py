"""
2026-09-25: EXPERT-PREFIX-label (not RHO-label) Request Pruner for bi400/ss80 at
CARDINALITY 3. Direct analog of build_request_pruner_bi200_ss100_c3_prefix.py /
build_request_pruner_bi400_ss200_c3_prefix.py - see build_request_pruner_bi200_ss100_c3_prefix.py's
docstring for the full reasoning. Depends on the c3 RHO-baseline collection
(run_rh_baseline_data_bi400_ss80_c3.sh) having completed - it has (2026-09-25, jobs
rh_c3_data_bi400_ss80_lc/lr/lrc all COMPLETED).

Run: python3 -m rtv_solver.pipeline.build_request_pruner_bi400_ss80_c3_prefix
"""
from pathlib import Path

from rtv_solver.pipeline.build_request_pruner_bi200_ss100_c3_prefix import build_prefix_dataset
from rtv_solver.pipeline.request_pruner_dataset_builder import build_dataset
from rtv_solver.pipeline.sweep_request_pruner import run_sweep

RH_BASELINE_DIR = Path("outputs/eval_rh_no_learning_bi400_ss80_c3")
SCAFFOLD_DIR = Path("dataset/request_pruning_bi400_ss80_c3")
PREFIX_DIR = Path("dataset/request_pruning_bi400_ss80_c3_prefix")
MODEL_OUTPUT_DIR = Path("outputs/request_pruner_mlp_bi400_ss80_c3_prefix")
K = 3  # = max_cardinality of the c3 SIL runs


if __name__ == "__main__":
    print(f"--- [1/3] building RHO scaffold dataset {RH_BASELINE_DIR} -> {SCAFFOLD_DIR} ---")
    build_dataset(rh_baseline_dir=RH_BASELINE_DIR, output_dir=SCAFFOLD_DIR)
    print(f"\n--- [2/3] swapping to expert-prefix labels (K={K}) {SCAFFOLD_DIR} -> {PREFIX_DIR} ---")
    build_prefix_dataset(SCAFFOLD_DIR, PREFIX_DIR, k=K)
    print(f"\n--- [3/3] sweeping hyperparameters -> {MODEL_OUTPUT_DIR} ---")
    run_sweep(dataset_dir=PREFIX_DIR, output_dir=MODEL_OUTPUT_DIR)
