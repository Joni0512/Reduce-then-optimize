"""
train_request_pruner.py
========================
2026-07-14: Phase 1 (training) of the request-pruner design discussion.

Trains RequestPrunerMLP on the dataset built by request_pruner_dataset_builder.py
(dataset/request_pruning/{train,val,test}/*.npz - raw, unnormalized features,
one row per (rolling-horizon window, active request)).

This first version runs exactly ONE hyperparameter configuration end to end
(hidden_dim=32, num_hidden_layers=2, dropout=0.1, pos_weight=1.5 - the
defaults discussed and agreed on 2026-07-14) to verify the whole pipeline
(load dataset -> normalize -> train -> evaluate -> checkpoint) works before
committing to the larger hyperparameter sweep (hidden_dim x num_hidden_layers
x dropout x pos_weight) planned as the next step. run_one_config() below is
written so that sweep can just call it in a loop later without restructuring
anything.

Conventions deliberately mirrored from the pair-pruner's training script
(train_models_v2.py) so both trainings are easy to compare:
    - Adam optimizer, ReduceLROnPlateau(mode="max") on val F3
    - best checkpoint = highest val F3 at a fixed threshold (0.5 for now)
    - early stopping via `patience`
    - metadata.json saved next to the checkpoint with every hyperparameter
Differences, and why:
    - No per-instance loop / balanced_indices subsampling: the pair pruner
      trains a GNN that needs one full graph per instance per forward pass.
      This MLP has no graph structure - every row is independent - so all
      train rows across all instances are just one big tensor, trained
      full-batch. Much simpler, and dataset is small enough (6.9k rows) that
      full-batch Adam for 100 epochs runs in seconds.
    - No balanced sampling: positive rate here is ~38% (see dataset
      manifest.json), nowhere near as skewed as the pair pruner's edge-level
      labels (single-digit %), so pos_weight alone is enough - matches the
      "don't just copy pos_weight values across, check the label ratio
      first" note from the original design discussion.

Run:
    python3 -m rtv_solver.pipeline.train_request_pruner
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rtv_solver.pipeline.request_pruner_dataset_builder import DEPLOYABLE_FEATURE_NAMES
from rtv_solver.pipeline.request_pruner_mlp import RequestPrunerMLP

# 2026-07-14: train on DEPLOYABLE_FEATURE_NAMES only, NOT the full
# FEATURE_NAMES stored in the dataset - the last 2 stored columns
# (num_idle_vehicles_proxy, requests_per_idle_vehicle) are computed from the
# solver's own output for that window (see the long comment in
# request_pruner_dataset_builder.py for how this was caught) and can't be
# computed before the pruner needs to run. Training on them would silently
# produce a checkpoint that RequestPruner could never actually use at
# inference time.
NUM_DEPLOYABLE_FEATURES = len(DEPLOYABLE_FEATURE_NAMES)


DATASET_DIR = Path("dataset/request_pruning")
OUTPUT_DIR = Path("outputs/request_pruner_mlp")
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(split: str, dataset_dir: Path = DATASET_DIR) -> dict:
    """
    Concatenates every instance's .npz file in <dataset_dir>/{split}/ into
    one flat set of arrays. No per-instance structure is needed here (unlike
    the pair pruner's per-instance graphs) since every row already stands
    alone - the graph/window structure was only needed to COMPUTE the
    features (done once, by the dataset builder), not to train this model.

    2026-07-15: dataset_dir made a parameter (default unchanged) so the same
    training code can be pointed at a window-config-specific dataset (see
    build_window_specific_request_pruners.py) instead of only the original
    400/100 dataset.
    """
    split_dir = dataset_dir / split
    feature_chunks, label_chunks, instance_chunks = [], [], []

    for npz_path in sorted(split_dir.glob("*.npz")):
        data = np.load(npz_path)
        # Slice down to the first NUM_DEPLOYABLE_FEATURES columns, dropping
        # the stored-but-not-deployable idle-vehicle-proxy columns - see the
        # NUM_DEPLOYABLE_FEATURES comment above and the long explanation in
        # request_pruner_dataset_builder.py.
        feature_chunks.append(data["features"][:, :NUM_DEPLOYABLE_FEATURES])
        label_chunks.append(data["label"])
        instance_chunks.append(np.full(len(data["label"]), npz_path.stem))

    return {
        "features": np.concatenate(feature_chunks, axis=0).astype(np.float32),
        "labels": np.concatenate(label_chunks, axis=0).astype(np.float32),
        "instances": np.concatenate(instance_chunks, axis=0),
    }


# ---------------------------------------------------------------------------
# Metrics - same small hand-rolled set used in request_pruner_signal_check.py
# (scikit-learn is not installed in this project's venv). Duplicated here
# rather than imported since that script is a throwaway analysis tool and
# this one is meant to become the permanent training entrypoint.
# ---------------------------------------------------------------------------


def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    precision_at_k = np.cumsum(labels_sorted) / (np.arange(len(labels_sorted)) + 1)
    return float((precision_at_k * labels_sorted).sum() / n_pos)


@torch.no_grad()
def evaluate(model: nn.Module, features: torch.Tensor, labels: np.ndarray, threshold: float) -> dict:
    model.eval()
    scores = torch.sigmoid(model(features)).numpy()
    preds = scores >= threshold

    tp = int((preds & (labels == 1)).sum())
    fp = int((preds & (labels == 0)).sum())
    fn = int((~preds & (labels == 1)).sum())
    tn = int((~preds & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": average_precision(labels, scores),
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1),
        "f2": f_beta(precision, recall, 2),
        "f3": f_beta(precision, recall, 3),
        "retention": float(preds.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_all_splits(dataset_dir: Path = DATASET_DIR) -> tuple[dict, dict, dict]:
    """
    Loads train/val/test once. Split out from run_one_config() so a sweep
    over many hyperparameter configs (see sweep_request_pruner.py) can load
    the ~10.6k rows from disk a single time instead of once per config -
    with 108 configs that would otherwise mean 108x re-reading the same 56
    .npz files for no reason, since the raw data never changes between
    configs (only normalization does, and that's derived from these same
    arrays each time anyway).
    """
    train = load_split("train", dataset_dir)
    val = load_split("val", dataset_dir)
    test = load_split("test", dataset_dir)
    print(
        f"train={len(train['labels'])} rows, val={len(val['labels'])} rows, "
        f"test={len(test['labels'])} rows"
    )
    return train, val, test


def train_one_config(
    train: dict,
    val: dict,
    test: dict,
    hidden_dim: int = 32,
    num_hidden_layers: int = 2,
    dropout: float = 0.1,
    pos_weight: float = 1.5,
    epochs: int = 100,
    learning_rate: float = 0.01,
    patience: int = 25,
    seed: int = 42,
    verbose: bool = True,
    save_checkpoint: bool = True,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    2026-07-15: output_dir made a parameter (default unchanged) so a
    window-config-specific model can be checkpointed under its own folder
    (e.g. outputs/request_pruner_mlp_bi200_ss100/) instead of overwriting the
    original 400/100 checkpoints under outputs/request_pruner_mlp/.

    Trains and evaluates exactly one hyperparameter configuration on
    already-loaded splits (see load_all_splits()). Used both by the
    single-config entry point below (verbose=True, full per-epoch/per-
    threshold printing) and by sweep_request_pruner.py (verbose=False, one
    line per config - 108 configs x full printing would be unreadable).

    Returns a self-contained summary dict (hyperparams + best epoch/F3 +
    final metrics at every threshold for every split) so a sweep can build
    its leaderboard purely from return values, without parsing stdout.

    save_checkpoint=False (2026-07-14, added for the 5-seed sweep): skips all
    disk writes (checkpoint .pt + metadata.json) and keeps the best epoch's
    weights in memory instead via best_state_dict. Needed because the 5-seed
    sweep runs 108 configs x 5 seeds = 540 trainings - saving a checkpoint
    for every one of those would mean 540 files on disk (most immediately
    discarded) AND, worse, every seed for the SAME hyperparameter combo would
    overwrite the same run_name's checkpoint path, silently leaving only the
    last seed's weights behind under a filename that no longer indicates
    which seed produced them. The sweep instead re-trains and saves only the
    actual winning config once, with save_checkpoint=True (see
    sweep_request_pruner.py).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # Train-set-fixed normalization (ddof=0, same reasoning as the "why
    # ddof=0" comment in request_pruner_signal_check.py: some rolling-horizon
    # windows have very few active requests, and computing mean/std per-split
    # here anyway is a non-issue since we normalize ALL splits with the SAME
    # train statistics - val/test never influence normalization, which is
    # the whole point of doing this at train time instead of per-window like
    # the pair pruner currently does at inference).
    feature_mean = train["features"].mean(axis=0)
    feature_std = train["features"].std(axis=0, ddof=0) + 1e-8

    def normalize(features: np.ndarray) -> torch.Tensor:
        return torch.tensor((features - feature_mean) / feature_std, dtype=torch.float32)

    train_x = normalize(train["features"])
    val_x = normalize(val["features"])
    test_x = normalize(test["features"])
    train_y = torch.tensor(train["labels"], dtype=torch.float32)

    model = RequestPrunerMLP(
        feature_dim=train_x.shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(3, patience // 3))

    # 2026-07-15: run_name now also encodes learning_rate - previously fixed
    # at 0.01 and omitted from the name, but the sweep now varies it too, and
    # without this two different learning rates for the same hidden_dim/
    # num_hidden_layers/dropout/pos_weight would silently collide on the same
    # checkpoint path.
    run_name = f"request_pruner_mlp_h{hidden_dim}_l{num_hidden_layers}_d{dropout}_pw{pos_weight}_lr{learning_rate}".replace(".", "p")
    run_dir = output_dir / run_name
    best_model_path = run_dir / f"{run_name}_best_val_f3.pt"

    best_val_f3 = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    selection_threshold = 0.5  # fixed for this first single-config run, same default as train_models_v2.py

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, val_x, val["labels"], selection_threshold)
        val_f3 = val_metrics["f3"]
        scheduler.step(val_f3)

        improved = val_f3 > best_val_f3 + 1e-8
        if improved:
            best_val_f3 = val_f3
            best_epoch = epoch
            epochs_without_improvement = 0
            # Always keep the best weights in memory (cheap - this is a tiny
            # MLP), independent of whether they also get written to disk.
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            if save_checkpoint:
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state_dict,
                        "feature_names": DEPLOYABLE_FEATURE_NAMES,
                        "feature_mean": feature_mean.tolist(),
                        "feature_std": feature_std.tolist(),
                        "hidden_dim": hidden_dim,
                        "num_hidden_layers": num_hidden_layers,
                        "dropout": dropout,
                        "pos_weight": pos_weight,
                        "threshold": selection_threshold,
                        "model_type": "request_pruner_mlp_v1",
                    },
                    best_model_path,
                )
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == epochs - 1 or improved:
            log(
                f"  epoch {epoch:3d}  loss={loss.item():.4f}  "
                f"val_f3={val_f3:.4f}  val_recall={val_metrics['recall']:.4f}  "
                f"val_precision={val_metrics['precision']:.4f}"
                + ("  * best" if improved else "")
            )

        if patience > 0 and epochs_without_improvement >= patience:
            log(f"  early stopping at epoch {epoch} (no val F3 improvement for {patience} epochs)")
            break

    train_time = time.time() - t0
    log(f"\nBest epoch: {best_epoch} (val_f3={best_val_f3:.4f}), trained in {train_time:.1f}s")

    # Restore the best epoch's weights from memory (not a disk reload - see
    # save_checkpoint docstring above) so printed/returned numbers match the
    # best epoch, not whatever the last epoch happened to be if it wasn't an
    # improvement.
    model.load_state_dict(best_state_dict)

    log("\n=== Final evaluation (best checkpoint) ===")
    # Full metrics[split][threshold] grid, not just the selection threshold -
    # a sweep leaderboard ranks by val_f3 at the fixed threshold, but keeping
    # every threshold's numbers around means the winning config's full
    # threshold/retention tradeoff is already on hand afterwards, no rerun
    # needed.
    metrics: dict[str, dict[float, dict]] = {}
    for split_name, x, labels in (("train", train_x, train["labels"]), ("val", val_x, val["labels"]), ("test", test_x, test["labels"])):
        log(f"\n  -- {split_name} --")
        metrics[split_name] = {}
        for threshold in THRESHOLDS:
            m = evaluate(model, x, labels, threshold)
            metrics[split_name][threshold] = m
            log(
                f"    threshold={threshold:.1f}  roc_auc={m['roc_auc']:.3f}  pr_auc={m['pr_auc']:.3f}  "
                f"recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
                f"f3={m['f3']:.3f}  retention={m['retention']:.3f}"
            )

    if save_checkpoint:
        metadata = {
            "run_name": run_name,
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "dropout": dropout,
            "pos_weight": pos_weight,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_f3": best_val_f3,
            "selection_threshold": selection_threshold,
            "train_time_seconds": train_time,
            "feature_names": DEPLOYABLE_FEATURE_NAMES,
            "best_model_path": str(best_model_path),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log(f"\nCheckpoint: {best_model_path}")
        log(f"Metadata:   {run_dir / 'metadata.json'}")

    return {
        "run_name": run_name,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "dropout": dropout,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_f3": best_val_f3,
        "selection_threshold": selection_threshold,
        "train_time_seconds": train_time,
        "best_model_path": str(best_model_path),
        "metrics": metrics,
    }


def run_one_config(dataset_dir: Path = DATASET_DIR, **kwargs) -> dict:
    """Convenience entry point for a single config: loads data once, trains, verbose by default."""
    train, val, test = load_all_splits(dataset_dir)
    return train_one_config(train, val, test, verbose=True, **kwargs)


if __name__ == "__main__":
    run_one_config()
