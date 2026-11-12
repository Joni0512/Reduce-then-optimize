"""
2026-11-08: NYC pair pruner trained on WINDOW graphs (overlap label) for RHO batch_interval=1800/step_size=900.

Reason: rule "one pruner per batch_interval/step_size" - the pruner of sweep_nyc_pair_pruner_window_rho3060.py
(window 3600/1800, labels from the 3600/1800 expert manifest) must not be reused at 1800/900 (shorter windows -> other
graph sizes, other per-window z-scores, other solution). Everything else is IDENTICAL to that script (it is imported):
window replication of OfflineRTVSolver.solve_rtv, per-window z-score, geographic features, hidden_dim=64, 100 epochs,
balanced sampling, BCEWithLogits(pos_weight), Adam, selection by val ROC-AUC. Differences only:
  * INTERVAL, STEP = 1800, 900 (patched into the imported module; build_window_dataset reads them at call time)
  * labels = overlap pairs of the plain RHO solution AT 1800/900 (no pruner, seed 1, cluster):
    outputs/outputs/new_tests/nyc_rho_baseline_bi1800_ss900/{train,val}_seed1/*/*/final/result_driver_runs.json
    (Train 991/1000 served, 873 overlap pairs; Val 1040/1047 served, 935 pairs). That file has requests, depot, driver_runs
    and travel_time_matrix, so it loads like the expert manifests. The "served -> leaves after first window" approximation
    uses the served set of THIS solution.
  * output: outputs/pair_pruner_window_bi1800_ss900/ with run names suffixed _bi1800_ss900.
Train day = 2016-01-12, val day = 2016-01-13 (checkpoint selection only), Test day (2016-01-19) untouched.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import rtv_solver.pipeline.sweep_nyc_pair_pruner_window_rho3060 as base
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import HIDDEN_DIM, EPOCHS, balanced_train_indices

base.INTERVAL, base.STEP = 1800, 900
OUT_DIR = Path("outputs/pair_pruner_window_bi1800_ss900")
RUN_GLOB = "outputs/outputs/new_tests/nyc_rho_baseline_bi1800_ss900/{split}_seed1/*/*/final/result_driver_runs.json"


def solution_path(split: str) -> Path:
    files = sorted(glob.glob(RUN_GLOB.format(split=split)))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly 1 baseline result for {split}, found {files}")
    return Path(files[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--pos_weight", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    a = p.parse_args()

    run_name = f"L{a.layers}_pw{a.pos_weight:g}_lr{a.lr:g}_seed{a.seed}_window_bi1800_ss900"
    print(f"=== {run_name} ===", flush=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    train_path, val_path = solution_path("train"), solution_path("val")
    t0 = time.time()
    train_ds, val_ds = base.build_window_dataset(train_path), base.build_window_dataset(val_path)
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
        logits = base.forward(model, train_ds)
        idx = balanced_train_indices(train_ds["labels"])
        loss = criterion(logits[idx], train_ds["labels"][idx])
        loss.backward()
        optimizer.step()
        if epoch % base.EVAL_EVERY == 0 or epoch == a.epochs - 1:
            m = base.evaluate(model, val_ds)
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
            "batch_interval": base.INTERVAL, "step_size": base.STEP, "label_mode": "overlap", "graph": "window",
            "train": str(train_path), "val": str(val_path),
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
