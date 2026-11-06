"""
2026-11-04: step 1 of the pair-pruner check - does the whole-day-trained GNN (overlap label) still work on
the window graphs the solver actually feeds it?

Reason: training normalizes edge/node features with mean/std over the whole-day graph (1000 requests), but
RequestGraphPruner.build_allowed_pairs normalizes per solver window (request_graph_pruner.py lines 116-122).
Time differences inside a 60-min window are much smaller than over an 8-h day, so z-scores differ.

Windows replicate OfflineRTVSolver.solve_rtv (batch_interval=3600, step_size=1800): a request is in the window
at current_time if pickup_window_end > t and pickup_window_start < t + 3600. APPROXIMATION: requests that the
expert (RHO 30/60) served are dropped after their first window (in the real run they leave `selected_requests`
once committed to a manifest); unserved requests stay until their pickup window ends. Labels = whole-day overlap
pairs of the expert manifest restricted to the pairs inside the window graph.

Compared variants (same checkpoint, same window graphs):
  window-norm : features z-scored per window (what the solver does today)
  day-norm    : features z-scored with the whole-day mean/std (what training used)
and the trivial time filter (-pickup_time_difference) as reference.
"""
import sys

import numpy as np
import pandas as pd
import torch

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder as FB
from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder
from rtv_solver.pipeline.sweep_nyc_pair_pruner_rho3060 import roc_auc
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import HIDDEN_DIM, VAL_PATH
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/pair_pruner_overlap_rho3060/L1_pw1_lr0.01_seed1_overlap_best_val_auc.pt"
LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
INTERVAL, STEP = 3600, 1800

config = Config()
payload = PayloadParser.load_input_data(VAL_PATH)
NetworkHandler.init_from_payload(payload=payload, server_url=config.SERVER_URL)
requests = RequestHandler(payload[PayloadKeys.REQUESTS], config).get_all_requests()

# whole-day normalization statistics (as in training)
G_day = FB.add_features(RequestGraphFullBuilder.build(requests), geographic=True)
nm, _, em, _ = FB.to_numpy(G_day)
node_df_day, edge_df_day = pd.DataFrame(nm, columns=FB.NODE_FEATURES), pd.DataFrame(em, columns=FB.EDGE_FEATURES)
node_mu, node_sd = node_df_day.mean(), node_df_day.std() + 1e-8
edge_mu, edge_sd = edge_df_day.mean(), edge_df_day.std() + 1e-8

pos_pairs = RequestGraphLabelBuilder.build_positive_pairs_overlap_from_solution_payload(payload)
served = {int(s["booking_id"]) for dr in payload["driver_runs"] for s in dr["manifest"] if int(s["booking_id"]) >= 0}

starts = [r.earliest_pickup_time for r in requests]
ends = [r.latest_pickup_time for r in requests]
t = max(0, min(starts) - INTERVAL)
t_end = max(ends)

model = RequestGraphEdgeGNN(
    node_feature_dim=len(FB.NODE_FEATURES), edge_feature_dim=len(FB.EDGE_FEATURES),
    hidden_dim=HIDDEN_DIM, message_passing_steps=LAYERS,
)
model.load_state_dict(torch.load(CKPT))
model.eval()
k_time = FB.EDGE_FEATURES.index("pickup_time_difference")

def score(node_df, edge_df, edge_index, node_ids, norm):
    if norm == "window":
        x = (node_df - node_df.mean()) / (node_df.std() + 1e-8)
        ea = (edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)
    else:
        x, ea = (node_df - node_mu) / node_sd, (edge_df - edge_mu) / edge_sd
    idx = {n: i for i, n in enumerate(node_ids)}
    ei = torch.tensor([[idx[u], idx[v]] for u, v in edge_index], dtype=torch.long).t().contiguous()
    with torch.no_grad():
        logit = model(x=torch.tensor(x.to_numpy(), dtype=torch.float32), edge_index=ei,
                      edge_attr=torch.tensor(ea.to_numpy(), dtype=torch.float32))
    return logit.numpy().astype(np.float64)

gone = set()  # served requests already committed in an earlier window
rows = {"window": [], "day": [], "time": []}  # per window: (scores, labels)
all_y, per_window = [], []
n_win = 0
while t < t_end:
    win = [r for r in requests if r.id not in gone and r.latest_pickup_time > t and r.earliest_pickup_time < t + INTERVAL]
    gone |= {r.id for r in win if r.id in served}
    t += STEP
    if len(win) <= 1:
        continue
    G = FB.add_features(RequestGraphFullBuilder.build(win), geographic=True)
    nm, ei, em, nid = FB.to_numpy(G)
    node_df, edge_df = pd.DataFrame(nm, columns=FB.NODE_FEATURES), pd.DataFrame(em, columns=FB.EDGE_FEATURES)
    y = np.array([1.0 if tuple(sorted((int(u), int(v)))) in pos_pairs else 0.0 for u, v in ei])
    s_win, s_day = score(node_df, edge_df, ei, nid, "window"), score(node_df, edge_df, ei, nid, "day")
    s_time = -em[:, k_time].astype(np.float64)
    for name, s in (("window", s_win), ("day", s_day), ("time", s_time)):
        rows[name].append(s)
    all_y.append(y)
    per_window.append((len(win), len(y), int(y.sum())))
    n_win += 1

y = np.concatenate(all_y)
n_pos = int(y.sum())
print(f"checkpoint {CKPT}\nwindows with >=2 requests: {n_win}, window edges total: {len(y):,}, positives: {n_pos}")
print(f"(whole-day positives {len(pos_pairs)}; positives not inside any window graph: {len(pos_pairs) - n_pos} - "
      "pairs that never co-occur in a window or involve a dropped request)\n")

def per_window_auc(name):
    aucs = []
    for s, yy in zip(rows[name], all_y):
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc(s, yy))
    return np.mean(aucs), len(aucs)

# pruning rate on the pooled set via a global score threshold would hide per-window effects; use the solver rule:
# probability >= 0.5 per window, i.e. logit >= 0
print(f"{'variant':<12}{'pooled AUC':>12}{'mean window AUC':>18}{'kept@0.5':>12}{'kept %':>9}{'recall@0.5':>12}")
for name, label in (("window", "window-norm"), ("day", "day-norm")):
    s = np.concatenate(rows[name])
    kept = s >= 0.0
    wa, nw = per_window_auc(name)
    print(f"{label:<12}{roc_auc(s, y):>12.4f}{wa:>14.4f} (n={nw}){int(kept.sum()):>9,}{100 * kept.mean():>9.2f}"
          f"{y[kept].sum() / n_pos:>12.3f}")
s = np.concatenate(rows["time"])
wa, nw = per_window_auc("time")
print(f"{'time filter':<12}{roc_auc(s, y):>12.4f}{wa:>14.4f} (n={nw})")

# recall of the time filter / GNNs when each keeps the same number of edges as the window-norm GNN at 0.5
keep_n = int((np.concatenate(rows["window"]) >= 0).sum())
print(f"\nrecall when every variant keeps the same number of edges (= {keep_n:,}, pooled ranking):")
for name in ("window", "day", "time"):
    sc = np.concatenate(rows[name])
    top = np.argsort(-sc, kind="mergesort")[:keep_n]
    print(f"  {name:<8} recall {y[top].sum() / n_pos:.3f}")
