"""
2026-09-22: Expert-prefix-label Request Pruners for bi400/ss80 and bi400/ss200, extending
the bi200/ss100 result (build_request_pruner_prefix_dataset.py /
train_eval_request_pruner_prefix_labels.py: real SIL training with the prefix-label pruner
recovered 66% of the RHO-label pruner's damage, 4/5 seeds, 2026-09-22).

Builds the label-swap datasets (same rows/features as the existing RHO-label datasets, only
`label` replaced) and trains one pruner per config with the SAME architecture the RHO-label
HP sweep already found best for that window config (build_window_specific_request_pruners.py),
only pos_weight forced to 1.0 - the bi200/ss100 test showed pos_weight 2/3 degenerate (F3
checkpoint selection picks a predict-all-keep epoch) and a fresh sweep for the prefix-label
objective is future work, not done here.

Run: python -m rtv_solver.pipeline.build_request_pruner_prefix_bi400
"""
import json
from pathlib import Path

import numpy as np

from rtv_solver.pipeline.build_request_pruner_prefix_dataset import K, prefix_labels
from rtv_solver.pipeline.train_request_pruner import load_all_splits, train_one_config

CONFIGS = [
    # (label, src_dataset_dir, dst_dataset_dir, out_dir, hidden_dim, num_hidden_layers, dropout, learning_rate)
    ("bi400_ss80", Path("dataset/request_pruning_bi400_ss80"), Path("dataset/request_pruning_bi400_ss80_prefix"),
     Path("outputs/request_pruner_mlp_bi400_ss80_prefix"), 64, 3, 0.1, 0.03),
    ("bi400_ss200", Path("dataset/request_pruning_bi400_ss200"), Path("dataset/request_pruning_bi400_ss200_prefix"),
     Path("outputs/request_pruner_mlp_bi400_ss200_prefix"), 32, 1, 0.0, 0.03),
]


def build_prefix_dataset(src_dir: Path, dst_dir: Path) -> None:
    manifest = json.load(open(src_dir / "manifest.json"))
    manifest["label_definition"] = (
        f"1.0 if the request is among the first K={K} requests of any vehicle's reference route "
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
            new = prefix_labels(z["node_ids"], z["window_index"], p.stem)
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


def main():
    for label, src, dst, out_dir, hidden_dim, num_layers, dropout, lr in CONFIGS:
        print(f"\n=== {label} ===")
        print(f"--- building prefix-label dataset {src} -> {dst}")
        build_prefix_dataset(src, dst)

        print(f"--- training pruner (h{hidden_dim} l{num_layers} d{dropout} pos_weight=1.0 lr{lr}) -> {out_dir}")
        train, val, test = load_all_splits(dst)
        res = train_one_config(
            train, val, test, hidden_dim=hidden_dim, num_hidden_layers=num_layers, dropout=dropout,
            pos_weight=1.0, learning_rate=lr, seed=42, verbose=False,
            save_checkpoint=True, output_dir=out_dir,
        )
        print(f"  best_val_f3={res['best_val_f3']:.3f} -> {res['best_model_path']}")


if __name__ == "__main__":
    main()
