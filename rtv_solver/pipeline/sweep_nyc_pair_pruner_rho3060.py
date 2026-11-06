"""
2026-11-01: Hyperparameter sweep for the NYC pair pruner trained on the RHO
batch_interval=3600/step_size=1800 (cardinality 3, seed 1) expert labels.

Reason: the first run (train_nyc_pair_pruner_gnn_rho3060.py, 1 layer, pw=1,
lr=1e-2) did not converge (loss ~0.60, val F3 0.184, precision 0.024 vs. base
rate 0.0175, strongly oscillating val kept_edges).

One call = one config; a SLURM array maps --task_id 0..26 onto the grid
layers x pos_weight x lr (see submit_pair_pruner_sweep_rho3060.sbatch).
Everything else is fixed as in the first run (hidden_dim=64, BALANCED=True,
100 epochs, same train/val manifests, Test day untouched).

Model selection is by val ROC-AUC (threshold independent), NOT by F3 at a
fixed threshold: pos_weight shifts the score distribution, so F3@0.5 is not
comparable across pos_weight values. Additionally reported at the best-AUC
epoch: recall of the top-50%-scored val edges (i.e. at a fixed 50% pruning
rate) and F3@0.5 for reference.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.train_nyc_pair_pruner_gnn_rho3060 import (
    TRAIN_PATH,
    VAL_PATH,
    HIDDEN_DIM,
    EPOCHS,
    build_graph_dataset,
    balanced_train_indices,
    evaluate_dataset,
)

LAYERS = [1, 2, 3]
POS_WEIGHTS = [1.0, 5.0, 20.0]
LEARNING_RATES = [1e-2, 3e-3, 1e-3]
GRID = list(itertools.product(LAYERS, POS_WEIGHTS, LEARNING_RATES))  # 27 configs

OUT_DIR = Path("outputs/pair_pruner_sweep_rho3060")
EVAL_EVERY = 5


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC; sklearn is not installed in this venv."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def val_scores(model, dataset):
    model.eval()
    with torch.no_grad():
        logits = model(
            x=dataset["x"],
            edge_index=dataset["edge_index"],
            edge_attr=dataset["edge_attr"],
        )
    return logits.numpy().astype(np.float64), dataset["labels"].numpy()


def recall_at_prune_rate(logits: np.ndarray, labels: np.ndarray, prune_rate: float) -> float:
    keep_n = int(round(len(logits) * (1.0 - prune_rate)))
    keep_idx = np.argsort(-logits, kind="mergesort")[:keep_n]
    return float(labels[keep_idx].sum() / labels.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None, help="0..26, index into the grid")
    parser.add_argument("--seed", type=int, default=1)
    # 2026-11-03: optional overrides for the long-training follow-up (best 2 configs of the
    # 100-epoch sweep had their val-AUC optimum at epoch 85-99, i.e. were not converged).
    # Without these flags the behaviour is identical to the 27-config sweep.
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    # 2026-11-03: "overlap" = positive only if both requests are on board the same vehicle at the
    # same time (default "vehicle" = served by the same vehicle at any time of the day).
    parser.add_argument("--label_mode", choices=["vehicle", "overlap"], default="vehicle")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    if args.task_id is not None:
        layers, pos_weight, lr = GRID[args.task_id]
    else:
        assert None not in (args.layers, args.pos_weight, args.lr), \
            "give --task_id or all of --layers --pos_weight --lr"
        layers, pos_weight, lr = args.layers, args.pos_weight, args.lr
    n_epochs = args.epochs
    out_dir = Path(args.out_dir)
    run_name = f"L{layers}_pw{pos_weight:g}_lr{lr:g}_seed{args.seed}"
    if n_epochs != EPOCHS:
        run_name += f"_ep{n_epochs}"
    if args.label_mode != "vehicle":
        run_name += f"_{args.label_mode}"
    print(f"=== {run_name} (task {args.task_id}, epochs {n_epochs}) ===", flush=True)

    # 2026-11-01: the first run was unseeded; seed here so replicates are reproducible.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    train_dataset = build_graph_dataset(TRAIN_PATH, label_mode=args.label_mode)
    val_dataset = build_graph_dataset(VAL_PATH, label_mode=args.label_mode)
    for name, ds in (("train", train_dataset), ("val", val_dataset)):
        pos = int(ds["labels"].sum().item())
        print(f"{name}: edges={ds['num_edges']} positives={pos} ({100 * pos / ds['num_edges']:.3f}%)", flush=True)
    t_build = time.time() - t0

    model = RequestGraphEdgeGNN(
        node_feature_dim=train_dataset["x"].shape[1],
        edge_feature_dim=train_dataset["edge_attr"].shape[1],
        hidden_dim=HIDDEN_DIM,
        message_passing_steps=layers,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}_best_val_auc.pt"

    best = {"val_auc": -1.0}
    history = []
    t1 = time.time()
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(
            x=train_dataset["x"],
            edge_index=train_dataset["edge_index"],
            edge_attr=train_dataset["edge_attr"],
        )
        train_idx = balanced_train_indices(train_dataset["labels"])
        loss = criterion(logits[train_idx], train_dataset["labels"][train_idx])
        loss.backward()
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == n_epochs - 1:
            v_logits, v_labels = val_scores(model, val_dataset)
            auc = roc_auc(v_logits, v_labels)
            rec50 = recall_at_prune_rate(v_logits, v_labels, 0.5)
            diag = evaluate_dataset(model, val_dataset)
            history.append({"epoch": epoch, "loss": loss.item(), "val_auc": auc,
                            "val_recall_at_50pct_pruned": rec50, "val_f3_at_0.5": diag["f_beta"]})
            print(f"epoch {epoch:03d} | loss {loss.item():.4f} | val AUC {auc:.4f} | "
                  f"val recall@50%pruned {rec50:.4f} | val F3@0.5 {diag['f_beta']:.4f}", flush=True)
            if auc > best["val_auc"]:
                best = {"epoch": epoch, "val_auc": auc, "val_recall_at_50pct_pruned": rec50,
                        "val_f3_at_0.5": diag["f_beta"], "val_precision_at_0.5": diag["precision"],
                        "val_recall_at_0.5": diag["recall"], "val_kept_edges_at_0.5": diag["kept_edges"]}
                torch.save(model.state_dict(), ckpt_path)

    result = {
        "run_name": run_name,
        "hyperparameters": {
            "message_passing_steps": layers, "pos_weight": pos_weight, "learning_rate": lr,
            "hidden_dim": HIDDEN_DIM, "epochs": n_epochs, "balanced": True, "seed": args.seed,
            "threshold_for_f3": 0.5, "train": str(TRAIN_PATH), "val": str(VAL_PATH),
            "label_mode": args.label_mode,
        },
        "best": best,
        "history": history,
        "runtime_s": {"graph_build": round(t_build, 1), "training": round(time.time() - t1, 1)},
    }
    (out_dir / f"{run_name}.json").write_text(json.dumps(result, indent=2))
    print(f"DONE {run_name} | best val AUC {best['val_auc']:.4f} @ epoch {best['epoch']} | "
          f"build {t_build:.0f}s train {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
