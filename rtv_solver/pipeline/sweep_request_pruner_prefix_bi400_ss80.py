"""
2026-09-22: HP sweep for the bi400/ss80 EXPERT-PREFIX-label Request Pruner, to fix the
mis-calibrated pruner from build_request_pruner_prefix_bi400.py.

Why: that script reused the RHO-label sweep's winning architecture (h64/l3/lr0.03)
unchanged for the prefix-label objective. Real SIL training with it made things WORSE than
the RHO-label pruner (service rate 36.5% vs 57.9%, i.e. -119% "recovered"), traced to a
badly calibrated model (best_epoch=3, test precision only 0.51 ~coin-flip, squashed score
range 0.24-0.89) - the deep net + lr=0.03 combo was likely too aggressive for the
differently-distributed prefix labels and stopped training too early via the F3-based
per-epoch checkpoint selection in train_one_config. By contrast bi200/ss100 and bi400/ss200
worked fine with the same reuse-architecture approach (bi400/ss200 even beat baseline).

This reuses sweep_request_pruner.py's run_sweep() UNCHANGED (only dataset_dir/output_dir
point at the new dataset) - its cross-config winner selection already ranks by
val_roc_auc_mean (threshold-independent, see that file's docstring on why F3-at-a-fixed-
threshold was replaced for exactly this degenerate-checkpoint reason), so a mis-calibrated
config should score low there and not win, without needing to touch shared code.
Full grid: HIDDEN_DIMS x NUM_HIDDEN_LAYERS x DROPOUTS x POS_WEIGHTS x LEARNING_RATES x
5 SEEDS = 432 configs x 5 seeds = 2160 trainings (same grid as the original RHO-label sweep).

Run: python -m rtv_solver.pipeline.sweep_request_pruner_prefix_bi400_ss80
"""
from pathlib import Path

from rtv_solver.pipeline.sweep_request_pruner import run_sweep

DATASET_DIR = Path("dataset/request_pruning_bi400_ss80_prefix")
OUTPUT_DIR = Path("outputs/request_pruner_mlp_bi400_ss80_prefix_sweep")

if __name__ == "__main__":
    run_sweep(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR)
