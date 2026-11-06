"""
2026-11-04: control for the NYC pair pruner with the "overlap" label. The GNN reaches val AUC 0.993
after epoch 0-ish; does a single raw edge feature already explain that (e.g. pickup_time_difference)?
For every edge feature the AUC of the raw feature as the only score is computed on the validation
(and train) day, no model involved. Sign-free: max(AUC, 1-AUC), with the direction reported.
Features are z-scored in build_graph_dataset, which does not change a rank-based AUC.
"""
import numpy as np

from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.sweep_nyc_pair_pruner_rho3060 import roc_auc
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import TRAIN_PATH, VAL_PATH, build_graph_dataset

for split, path in (("val", VAL_PATH), ("train", TRAIN_PATH)):
    ds = build_graph_dataset(path, label_mode="overlap")
    y = ds["labels"].numpy()
    X = ds["edge_attr"].numpy().astype(np.float64)
    print(f"\n=== {split}: edges={len(y)} positives={int(y.sum())} ===")
    rows = []
    for k, name in enumerate(RequestGraphFeatureBuilder.EDGE_FEATURES):
        a = roc_auc(X[:, k], y)
        rows.append((max(a, 1 - a), name, "higher=positive" if a >= 0.5 else "lower=positive"))
    for a, name, direction in sorted(rows, reverse=True):
        print(f"  {name:<36} AUC {a:.4f}  ({direction})")
