"""
2026-11-05: full hyperparameter sweep (sweep_request_pruner.run_sweep: 432 configs x 5 seeds) for ONE window-specific Request Pruner.
Same recipe as the 40/10 pruner (dataset built by request_pruner_dataset_builder.build_dataset from outputs/eval_rh_no_learning_bi<BI>_ss<SS>),
no change to the sweep itself. Window-specific per CLAUDE.md rule 2026-10-20 (never reuse another window's pruner).
Usage: python -m rtv_solver.pipeline.run_request_pruner_window_sweep <batch_interval> <step_size>
Input:  dataset/request_pruning_bi<BI>_ss<SS>/   Output: outputs/request_pruner_mlp_bi<BI>_ss<SS>/
"""
import sys
import time
from pathlib import Path

from rtv_solver.pipeline.sweep_request_pruner import run_sweep

if __name__ == "__main__":
    bi, ss = sys.argv[1], sys.argv[2]
    lab = f"bi{bi}_ss{ss}"
    ds = Path(f"dataset/request_pruning_{lab}")
    assert (ds / "manifest.json").exists(), f"dataset missing: {ds}"
    t0 = time.time()
    run_sweep(dataset_dir=ds, output_dir=Path(f"outputs/request_pruner_mlp_{lab}"))
    print(f"[{lab}] sweep done, wall-clock {time.time() - t0:.0f}s ({(time.time() - t0) / 3600:.2f} h)", flush=True)
