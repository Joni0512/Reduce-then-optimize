"""
2026-11-04: NYC pair pruner trained on WINDOW graphs (overlap label), RHO batch_interval=3600/step_size=1800.

Reason: the whole-day-trained pruner (sweep_nyc_pair_pruner_rho3060.py, --label_mode overlap) reaches val AUC
0.99 on the day graph, but on the graphs the solver really builds (one per solver call, features z-scored per
window in RequestGraphPruner.build_allowed_pairs) the same checkpoint drops to AUC 0.72 (see
scripts/analysis/pair_pruner_overlap_window_eval.py). So training now mirrors the solver:
  * windows replicate OfflineRTVSolver.solve_rtv: t starts at max(0, first_pickup_start - 3600), step 1800;
    request is in the window iff pickup_window_end > t and pickup_window_start < t + 3600
  * APPROXIMATION: requests served in the expert (RHO 30/60) manifest are removed after their first window
    (in the real run they leave `selected_requests` once committed); unserved requests stay until their
    pickup window ends
  * one request graph per window (RequestGraphFullBuilder + features with geographic=True), node/edge
    features z-scored PER WINDOW, exactly like the solver; windows are stacked into one disjoint graph
    (the GNN has no global op, so this equals running window by window)
  * label = overlap pair of the whole-day expert manifest (both requests on board the same vehicle at the same
    time), restricted to the pairs that exist in the window graph

Everything else as in sweep_nyc_pair_pruner_rho3060.py: hidden_dim=64, 100 epochs, BALANCED sampling (50:50
per epoch over all window edges), BCEWithLogits(pos_weight), Adam, selection by val ROC-AUC (evaluated every 5
epochs). Extra val metrics at the solver rule (probability >= 0.5, per window): kept share and recall.
Test day (2016-01-19) untouched.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder as FB
from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder
from rtv_solver.pipeline.sweep_nyc_pair_pruner_rho3060 import roc_auc, recall_at_prune_rate
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import (
    TRAIN_PATH, VAL_PATH, HIDDEN_DIM, EPOCHS, balanced_train_indices,
)
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

INTERVAL, STEP = 3600, 1800
OUT_DIR = Path("outputs/pair_pruner_window_rho3060")
EVAL_EVERY = 5


def build_window_dataset(payload_path: Path):
    config = Config()
    payload = PayloadParser.load_input_data(payload_path)
    NetworkHandler.init_from_payload(payload=payload, server_url=config.SERVER_URL)
    requests = RequestHandler(payload[PayloadKeys.REQUESTS], config).get_all_requests()

    pos_pairs = RequestGraphLabelBuilder.build_positive_pairs_overlap_from_solution_payload(payload)
    served = {int(s["booking_id"]) for dr in payload["driver_runs"] for s in dr["manifest"]
              if int(s["booking_id"]) >= 0}

    t = max(0, min(r.earliest_pickup_time for r in requests) - INTERVAL)
    t_end = max(r.latest_pickup_time for r in requests)
    gone = set()
    xs, eis, eas, ys, wid = [], [], [], [], []
    node_off, n_win = 0, 0
    while t < t_end:
        win = [r for r in requests if r.id not in gone
               and r.latest_pickup_time > t and r.earliest_pickup_time < t + INTERVAL]
        gone |= {r.id for r in win if r.id in served}
        t += STEP
        if len(win) <= 1:
            continue
        G = FB.add_features(RequestGraphFullBuilder.build(win), geographic=True)
        nm, ei, em, nid = FB.to_numpy(G)
        node_df, edge_df = pd.DataFrame(nm, columns=FB.NODE_FEATURES), pd.DataFrame(em, columns=FB.EDGE_FEATURES)
        # per-window z-score, identical to RequestGraphPruner.build_allowed_pairs
        x = ((node_df - node_df.mean()) / (node_df.std() + 1e-8)).to_numpy()
        ea = ((edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)).to_numpy()
        idx = {n: i for i, n in enumerate(nid)}
        xs.append(x)
        eis.append(np.array([[idx[u] + node_off, idx[v] + node_off] for u, v in ei]))
        eas.append(ea)
        ys.append(np.array([1.0 if tuple(sorted((int(u), int(v)))) in pos_pairs else 0.0 for u, v in ei]))
        wid.append(np.full(len(ei), n_win))
        node_off += len(nid)
        n_win += 1

    return {
        "name": payload_path.stem,
        "x": torch.tensor(np.concatenate(xs), dtype=torch.float32),
        "edge_index": torch.tensor(np.concatenate(eis), dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(np.concatenate(eas), dtype=torch.float32),
        "labels": torch.tensor(np.concatenate(ys), dtype=torch.float32),
        "window_id": np.concatenate(wid),
        "num_windows": n_win,
        "num_edges": int(sum(len(y) for y in ys)),
        "day_positives": len(pos_pairs),
    }


def forward(model, ds):
    return model(x=ds["x"], edge_index=ds["edge_index"], edge_attr=ds["edge_attr"])


def evaluate(model, ds):
    model.eval()
    with torch.no_grad():
        logits = forward(model, ds).numpy().astype(np.float64)
    y = ds["labels"].numpy()
    kept = logits >= 0.0  # solver rule: sigmoid(logit) >= 0.5
    n_pos = y.sum()
    aucs = []
    for w in range(ds["num_windows"]):
        m = ds["window_id"] == w
        if 0 < y[m].sum() < m.sum():
            aucs.append(roc_auc(logits[m], y[m]))
    return {
        "auc": roc_auc(logits, y),
        "mean_window_auc": float(np.mean(aucs)),
        "recall_at_50pct_pruned": recall_at_prune_rate(logits, y, 0.5),
        "recall_at_95pct_pruned": recall_at_prune_rate(logits, y, 0.95),
        "kept_share_at_0.5": float(kept.mean()),
        "recall_at_0.5": float(y[kept].sum() / n_pos),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--pos_weight", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    a = p.parse_args()

    run_name = f"L{a.layers}_pw{a.pos_weight:g}_lr{a.lr:g}_seed{a.seed}_window"
    print(f"=== {run_name} ===", flush=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    t0 = time.time()
    train_ds, val_ds = build_window_dataset(TRAIN_PATH), build_window_dataset(VAL_PATH)
    for n, ds in (("train", train_ds), ("val", val_ds)):
        pos = int(ds["labels"].sum())
        print(f"{n}: windows={ds['num_windows']} nodes={ds['x'].shape[0]} edges={ds['num_edges']} "
              f"positives={pos} ({100 * pos / ds['num_edges']:.3f}%) of {ds['day_positives']} day positives", flush=True)
    t_build = time.time() - t0

    model = RequestGraphEdgeGNN(
        node_feature_dim=train_ds["x"].shape[1], edge_feature_dim=train_ds["edge_attr"].shape[1],
        hidden_dim=HIDDEN_DIM, message_passing_steps=a.layers,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([a.pos_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{run_name}_best_val_auc.pt"
    best, history = {"val_auc": -1.0}, []
    t1 = time.time()
    for epoch in range(a.epochs):
        model.train()
        optimizer.zero_grad()
        logits = forward(model, train_ds)
        idx = balanced_train_indices(train_ds["labels"])
        loss = criterion(logits[idx], train_ds["labels"][idx])
        loss.backward()
        optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == a.epochs - 1:
            m = evaluate(model, val_ds)
            history.append({"epoch": epoch, "loss": loss.item(), **m})
            print(f"epoch {epoch:03d} | loss {loss.item():.4f} | val AUC {m['auc']:.4f} | "
                  f"win-AUC {m['mean_window_auc']:.4f} | kept@0.5 {m['kept_share_at_0.5']:.3f} | "
                  f"recall@0.5 {m['recall_at_0.5']:.3f} | recall@95%pruned {m['recall_at_95pct_pruned']:.3f}", flush=True)
            if m["auc"] > best["val_auc"]:
                best = {"epoch": epoch, "val_auc": m["auc"], **{f"val_{k}": v for k, v in m.items() if k != "auc"}}
                torch.save(model.state_dict(), ckpt)

    result = {
        "run_name": run_name,
        "hyperparameters": {
            "message_passing_steps": a.layers, "pos_weight": a.pos_weight, "learning_rate": a.lr,
            "hidden_dim": HIDDEN_DIM, "epochs": a.epochs, "balanced": True, "seed": a.seed,
            "batch_interval": INTERVAL, "step_size": STEP, "label_mode": "overlap", "graph": "window",
            "train": str(TRAIN_PATH), "val": str(VAL_PATH),
        },
        "data": {"train_windows": train_ds["num_windows"], "train_edges": train_ds["num_edges"],
                 "train_positives": int(train_ds["labels"].sum()), "val_windows": val_ds["num_windows"],
                 "val_edges": val_ds["num_edges"], "val_positives": int(val_ds["labels"].sum())},
        "best": best,
        "history": history,
        "runtime_s": {"graph_build": round(t_build, 1), "training": round(time.time() - t1, 1)},
    }
    (out_dir / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    print(f"DONE {run_name} | best val AUC {best['val_auc']:.4f} @ epoch {best['epoch']} | "
          f"build {t_build:.0f}s train {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
