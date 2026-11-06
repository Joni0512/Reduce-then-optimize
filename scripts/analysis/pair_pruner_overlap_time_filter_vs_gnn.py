"""
2026-11-04: fair comparison on the NYC validation day (overlap label): recall of the positive pairs
at equal pruning rates for (a) the trivial filter "keep the pairs with the smallest
pickup_time_difference" and (b) the trained pair-pruner GNN (layers=1, pos_weight=1, lr=1e-2,
seed 1, 100 epochs, checkpoint of best val AUC). Also: edges that must be kept to reach a target recall.
"""
import numpy as np
import torch

from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import HIDDEN_DIM, VAL_PATH, build_graph_dataset

CKPT = "outputs/pair_pruner_overlap_rho3060/L1_pw1_lr0.01_seed1_overlap_best_val_auc.pt"

ds = build_graph_dataset(VAL_PATH, label_mode="overlap")
y = ds["labels"].numpy()
n_edges, n_pos = len(y), int(y.sum())

model = RequestGraphEdgeGNN(
    node_feature_dim=ds["x"].shape[1], edge_feature_dim=ds["edge_attr"].shape[1],
    hidden_dim=HIDDEN_DIM, message_passing_steps=1,
)
model.load_state_dict(torch.load(CKPT))
model.eval()
with torch.no_grad():
    gnn_score = model(x=ds["x"], edge_index=ds["edge_index"], edge_attr=ds["edge_attr"]).numpy().astype(np.float64)

k = RequestGraphFeatureBuilder.EDGE_FEATURES.index("pickup_time_difference")
time_score = -ds["edge_attr"].numpy()[:, k].astype(np.float64)  # smaller time difference = higher score

scores = {"time filter (pickup_time_difference)": time_score, "GNN (L1, pw1, lr1e-2, seed 1)": gnn_score}
order = {n: np.argsort(-s, kind="mergesort") for n, s in scores.items()}
cum = {n: np.cumsum(y[o]) / n_pos for n, o in order.items()}

print(f"val edges={n_edges} positives={n_pos}\n")
print("recall of positive pairs at a given pruning rate (share of edges removed):")
rates = [0.50, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99]
print(f"{'':<40}" + "".join(f"{int(r * 100):>7}%" for r in rates))
for n in scores:
    row = "".join(f"{cum[n][int(round(n_edges * (1 - r))) - 1]:>8.3f}" for r in rates)
    print(f"{n:<40}{row}")

print("\nedges that must be kept (and % of all edges) to reach a target recall:")
for target in (0.90, 0.95, 0.99, 1.00):
    parts = []
    for n in scores:
        kept = int(np.searchsorted(cum[n], target - 1e-12) + 1)
        parts.append(f"{n}: {kept:,} ({100 * kept / n_edges:.2f}%)")
    print(f"  recall >= {target:.2f} | " + " | ".join(parts))
