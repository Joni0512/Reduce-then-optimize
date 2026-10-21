"""
2026-09-24: EXPERT-PREFIX-label (not RHO-label) Request Pruner for bi200/ss100 at
CARDINALITY 3 - the c3 continuation of the c2 fix (build_request_pruner_prefix_dataset.py,
build_request_pruner_prefix_bi400.py). Replaces build_request_pruner_bi200_ss100_c3.py,
which swept the OLD "assigned-in-window" RHO label unchanged - the label shown this
session to make legacy-scoring SIL training worse, not better (see chat 2026-09-21/22).

Three steps:
1. build_dataset() from the c3 RHO-baseline logs (run_rh_baseline_data_bi200_ss100_c3.sh)
   -> SCAFFOLD_DIR. This step's own RHO label is discarded; only the row/window/feature
   structure (which requests are candidates in which window, at max_cardinality=3) is reused.
2. Swap labels via prefix_labels(..., k=3) (K = max_cardinality here, NOT the module's
   default K=2 constant - build_request_pruner_prefix_bi400.py's build_prefix_dataset()
   silently used the default and would be wrong here) -> PREFIX_DIR.
3. Full HP sweep (same grid as sweep_request_pruner.py, ranked by val_roc_auc_mean - see
   sweep_request_pruner_prefix_bi400_ss80.py's docstring for why the grid is NOT
   shortcut/reused across configs) -> MODEL_OUTPUT_DIR.

Run: python3 -m rtv_solver.pipeline.build_request_pruner_bi200_ss100_c3_prefix
"""
import json
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.build_request_pruner_prefix_dataset import prefix_labels
from rtv_solver.pipeline.request_pruner_dataset_builder import build_dataset
from rtv_solver.pipeline.sweep_request_pruner import run_sweep

RH_BASELINE_DIR = Path("outputs/eval_rh_no_learning_bi200_ss100_c3")
SCAFFOLD_DIR = Path("dataset/request_pruning_bi200_ss100_c3")
PREFIX_DIR = Path("dataset/request_pruning_bi200_ss100_c3_prefix")
MODEL_OUTPUT_DIR = Path("outputs/request_pruner_mlp_bi200_ss100_c3_prefix")
K = 3  # = max_cardinality of the c3 SIL runs


def build_prefix_dataset(src_dir: Path, dst_dir: Path, k: int) -> None:
    manifest = json.load(open(src_dir / "manifest.json"))
    manifest["label_definition"] = (
        f"1.0 if the request is among the first K={k} requests of any vehicle's reference route "
        "projected onto this window's batch (what legacy imitation scoring can reward), else 0.0. "
        "Batches = RHO-baseline windows (exact in window 0, approximation afterwards). "
        f"Rows/features identical to {src_dir}."
    )
    manifest["source_dataset"] = str(src_dir)
    for split in ("train", "val", "test"):
        (dst_dir / split).mkdir(parents=True, exist_ok=True)
        pos_new = pos_old = rows = 0
        for p in sorted((src_dir / split).glob("*.npz")):
            z = np.load(p)
            new = prefix_labels(z["node_ids"], z["window_index"], p.stem, k=k)
            np.savez(
                dst_dir / split / p.name,
                node_ids=z["node_ids"], window_index=z["window_index"],
                timestamp=z["timestamp"], label=new, features=z["features"],
            )
            pos_new += new.sum(); pos_old += z["label"].sum(); rows += len(new)
            if p.stem in manifest["instances"]:
                manifest["instances"][p.stem]["positive_rate"] = float(new.mean())
        print(f"  {split:5s}: rows={rows:6d}  positive rate prefix={pos_new / rows:.3f}  (RHO label was {pos_old / rows:.3f})")
    json.dump(manifest, open(dst_dir / "manifest.json", "w"), indent=2)


if __name__ == "__main__":
    print(f"--- [1/3] building RHO scaffold dataset {RH_BASELINE_DIR} -> {SCAFFOLD_DIR} ---")
    build_dataset(rh_baseline_dir=RH_BASELINE_DIR, output_dir=SCAFFOLD_DIR)
    print(f"\n--- [2/3] swapping to expert-prefix labels (K={K}) {SCAFFOLD_DIR} -> {PREFIX_DIR} ---")
    build_prefix_dataset(SCAFFOLD_DIR, PREFIX_DIR, k=K)
    print(f"\n--- [3/3] sweeping hyperparameters -> {MODEL_OUTPUT_DIR} ---")
    run_sweep(dataset_dir=PREFIX_DIR, output_dir=MODEL_OUTPUT_DIR)
