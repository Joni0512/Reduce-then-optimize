"""
build_window_specific_request_pruners.py
==========================================
2026-07-15: window-size robustness for the request pruner, via SEPARATE
specialized models rather than one generalist model trained on merged data.

Background: the original request pruner (outputs/request_pruner_mlp/, see
train_request_pruner.py / sweep_request_pruner.py) was trained only on
BATCH_INTERVAL=400/STEP_SIZE=100 baseline logs. When deployed at a different
window config (BATCH_INTERVAL=200/STEP_SIZE=100), it collapsed on instances
with small fleets (lr211, lrc208): requests_per_vehicle reached values
(z-score up to +2.9) far outside anything seen during training, and the
single-hidden-layer MLP extrapolated by driving nearly all scores toward
zero instead of degrading gracefully - catastrophic over-pruning, up to
-16pp service rate on the worst instance.

Two fixes were discussed: (a) merge training data from multiple window
configs into one dataset and train one generalist model, or (b) train one
specialized model per window config. (b) was chosen (2026-07-15 design
discussion): the pruner is always deployed against ONE known window config
at a time (we test 100-200, 400-200, 50-10 individually, never
simultaneously), so there's no actual need for a single model to generalize
across configs - and (b) reuses the entire existing dataset-builder/train/
sweep pipeline unchanged (just pointed at different paths), instead of
building new data-merging logic.

This script is the driver for (b): for each of the 3 additional window
configs already simulated (see run_multiwindow_baseline_data.sh), it:
    1. builds a dataset from that config's baseline logs
       (outputs/eval_rh_no_learning_bi<BI>_ss<SS>/) into its own directory
       (dataset/request_pruning_bi<BI>_ss<SS>/)
    2. runs the full hyperparameter sweep (see sweep_request_pruner.py -
       now includes learning_rate too) against that dataset, saving the
       winning checkpoint under its own output directory
       (outputs/request_pruner_mlp_bi<BI>_ss<SS>/)

The original 400/100 dataset/model are never touched.

Run:
    python3 -m rtv_solver.pipeline.build_window_specific_request_pruners
"""

from __future__ import annotations

from pathlib import Path

from rtv_solver.pipeline.request_pruner_dataset_builder import build_dataset
from rtv_solver.pipeline.sweep_request_pruner import run_sweep


# (BATCH_INTERVAL, STEP_SIZE) pairs to build specialized models for - matches
# run_multiwindow_baseline_data.sh, which already generated the baseline logs
# these datasets are built from.
WINDOW_CONFIGS = [(200, 100), (400, 200), (50, 10)]


def build_all() -> None:
    for batch_interval, step_size in WINDOW_CONFIGS:
        label = f"bi{batch_interval}_ss{step_size}"
        rh_baseline_dir = Path(f"outputs/eval_rh_no_learning_{label}")
        dataset_dir = Path(f"dataset/request_pruning_{label}")
        model_output_dir = Path(f"outputs/request_pruner_mlp_{label}")

        print(f"\n{'=' * 80}")
        print(f"Window config: BATCH_INTERVAL={batch_interval}, STEP_SIZE={step_size}")
        print(f"{'=' * 80}")

        print(f"\n--- Building dataset from {rh_baseline_dir} -> {dataset_dir} ---")
        build_dataset(rh_baseline_dir=rh_baseline_dir, output_dir=dataset_dir)

        print(f"\n--- Sweeping hyperparameters -> {model_output_dir} ---")
        run_sweep(dataset_dir=dataset_dir, output_dir=model_output_dir)


if __name__ == "__main__":
    build_all()
