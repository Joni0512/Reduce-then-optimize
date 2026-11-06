"""
2026-11-05: train the NYC request pruner (MLP, train_request_pruner.train_one_config) on the dataset with
the label "pickup_before_next_resolve" (build_nyc_request_pruner_dataset_pickup_label.py), one config.

Test-day rule: that dataset has no test split (Test 2016-01-19 untouched). train_one_config needs a `test`
argument and prints metrics for it, so the VAL split is passed as a placeholder: every "test" line in the
output is just a duplicate of val and carries no information. Checkpoint selection is by val F3 at 0.5
(unchanged train_one_config behaviour); the offline check below does NOT use the selected threshold but
the label-independent ranking quality, because F3@0.5 is confounded by pos_weight.

After training: on the VAL split, for pruning the lowest-scored x % of ALL (window, request) rows, how many
label-1 rows are lost (global ranking), and the same with ranking inside each window (window-wise).

Run:
    python3 -m train_nyc_request_pruner_pickup_label --batch_interval 3600 --step_size 1800
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP
from rtv_solver.pipeline.train_request_pruner import load_split, train_one_config

PRUNE_RATES = (0.1, 0.2, 0.3, 0.4, 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_interval", type=int, required=True)
    p.add_argument("--step_size", type=int, required=True)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    tag = f"bi{a.batch_interval}_ss{a.step_size}"
    dataset_dir = Path(f"dataset/request_pruning_nyc_pickup_{tag}")
    output_dir = Path(f"outputs/request_pruner_mlp_nyc_pickup_{tag}")
    train, val = load_split("train", dataset_dir), load_split("val", dataset_dir)
    print(f"train={len(train['labels'])} rows (pos {train['labels'].mean():.3f}), "
          f"val={len(val['labels'])} rows (pos {val['labels'].mean():.3f})  [test = val placeholder]")

    res = train_one_config(
        train, val, val,  # val passed twice: the second one is the "test" placeholder, see docstring
        hidden_dim=a.hidden_dim, num_hidden_layers=a.num_hidden_layers, dropout=a.dropout,
        pos_weight=a.pos_weight, learning_rate=a.learning_rate, seed=a.seed,
        verbose=False, save_checkpoint=True, output_dir=output_dir,
    )
    print(f"\nrun {res['run_name']}  best epoch {res['best_epoch']}  best val F3@0.5 {res['best_val_f3']:.4f}")
    m = res["metrics"]["val"]
    for t, v in m.items():
        print(f"  val threshold {t:.1f}: roc_auc {v['roc_auc']:.3f} pr_auc {v['pr_auc']:.3f} "
              f"recall {v['recall']:.3f} precision {v['precision']:.3f} retention {v['retention']:.3f}")

    ckpt = torch.load(res["best_model_path"], map_location="cpu")
    model = RequestPrunerMLP(feature_dim=len(ckpt["feature_mean"]), hidden_dim=ckpt["hidden_dim"],
                             num_hidden_layers=ckpt["num_hidden_layers"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = torch.tensor((val["features"] - np.array(ckpt["feature_mean"])) / np.array(ckpt["feature_std"]),
                     dtype=torch.float32)
    with torch.no_grad():
        score = model(x).numpy().astype(np.float64)
    y = val["labels"]
    # load_split does not return window_index; the val split is one .npz file, same row order
    win = np.concatenate([np.load(f)["window_index"] for f in sorted((dataset_dir / "val").glob("*.npz"))])
    assert len(win) == len(y)
    n_pos = y.sum()
    print(f"\nVAL: label-1 rows lost when pruning the lowest-scored x % of rows (n_pos={int(n_pos)}):")
    print(f"{'prune %':>8}{'global lost':>14}{'window-wise lost':>20}")
    for r in PRUNE_RATES:
        keep = np.zeros(len(y), dtype=bool)
        keep[np.argsort(-score, kind="mergesort")[: int(round(len(y) * (1 - r)))]] = True
        ww = np.zeros(len(y), dtype=bool)
        for w in np.unique(win):
            idx = np.where(win == w)[0]
            ww[idx[np.argsort(-score[idx], kind="mergesort")[: int(round(len(idx) * (1 - r)))]]] = True
        print(f"{int(r * 100):>7}%{100 * (1 - y[keep].sum() / n_pos):>13.1f} %{100 * (1 - y[ww].sum() / n_pos):>19.1f} %")
    print("  (reference: pruning random rows loses the same share of label-1 rows as the prune rate)")


if __name__ == "__main__":
    main()
