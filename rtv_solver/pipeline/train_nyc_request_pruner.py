"""
2026-08-27: NYC request-pruner training (Phase 1, step 3 of the NYC
RequestPruner readiness plan). Thin wrapper around train_request_pruner.py's
run_one_config() - all the actual training/eval logic lives there and is
dataset_dir/output_dir-parameterized already, no duplication needed.

pos_weight=1.0 (not the Li&Lim default of 1.5): that default was tuned for
Li&Lim's ~38% positive rate (positives are the minority there, so upweighted).
NYC's positive rate at the 4min/8min config is ~78% (see
dataset/request_pruning_nyc/manifest.json) - positives are the MAJORITY here,
so reusing 1.5 would upweight the already-dominant class in the wrong
direction. Starting neutral (1.0) for this first verification run - see
train_request_pruner.py's own docstring note: "don't just copy pos_weight
values across, check the label ratio first".

Run:
    python3 -m rtv_solver.pipeline.train_nyc_request_pruner
"""
from pathlib import Path

from rtv_solver.pipeline.train_request_pruner import run_one_config

DATASET_DIR = Path("dataset/request_pruning_nyc")
OUTPUT_DIR = Path("outputs/request_pruner_mlp_nyc")

if __name__ == "__main__":
    run_one_config(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        pos_weight=1.0,
    )
