from __future__ import annotations

"""
train_models_v2.py
==================
Clean training pipeline for request-graph GNN pruning experiments.

What it does
------------
- trains class-specific and mixed request-graph GNN models
- runs a pos_weight loss sweep
- supports both GNN architectures via --gnn_version {v1,v2}, so the same sweep
  can be trained for each and compared head-to-head
- logs precision, recall, F1, F2, F3, F4, TP/FP/FN/TN
- saves best validation checkpoint per run
- stores everything in a new output folder so old models are not overwritten

Default output folder
---------------------
outputs/models_v2/      (--gnn_version v1, the default output_dir if unset)
outputs/models_v2_gnnv2/ (--gnn_version v2)

Example usage
-------------
Quick smoke test:
    python3 train_models_v2.py --models mixed_c1 --pos_weights 1 5 --epochs 5

Recommended first run (v2 architecture):
    python3 train_models_v2.py --models mixed_c1 mixed_c2 mixed_all --pos_weights 1 2 3 5 7.5 10

Same sweep with the older v1 architecture, for comparison:
    python3 train_models_v2.py --gnn_version v1 --models mixed_c1 mixed_c2 mixed_all --pos_weights 1 2 3 5 7.5 10

Full sweep:
    python3 train_models_v2.py --models lc1 lr1 lrc1 lc2 lr2 lrc2 mixed_c1 mixed_c2 mixed_all --pos_weights 1 2 3 5 7.5 10
"""

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
# 2026-07-07: script now trains either architecture via --gnn_version {v1,v2}, so
# the same sweep (splits, pos_weights, thresholds) can be run for both and compared
# head-to-head instead of only ever training the newer RequestGraphEdgeGNNv2.
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_gnn_v2 import RequestGraphEdgeGNNv2
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

SPLITS: dict[str, dict[str, list[str]]] = {
    "lc1": {
        "train": ["lc101", "lc102", "lc103", "lc104", "lc105", "lc106"],
        "val": ["lc107"],
        "test": ["lc108", "lc109"],
    },
    "lr1": {
        "train": ["lr101", "lr102", "lr103", "lr104", "lr105", "lr106", "lr107", "lr108"],
        "val": ["lr109", "lr110"],
        "test": ["lr111", "lr112"],
    },
    "lrc1": {
        "train": ["lrc101", "lrc102", "lrc103", "lrc104", "lrc105"],
        "val": ["lrc106"],
        "test": ["lrc107", "lrc108"],
    },
    "lc2": {
        "train": ["lc201", "lc202", "lc203", "lc204", "lc205"],
        "val": ["lc206"],
        "test": ["lc207", "lc208"],
    },
    "lr2": {
        "train": ["lr201", "lr202", "lr203", "lr204", "lr205", "lr206", "lr207"],
        "val": ["lr208", "lr209"],
        "test": ["lr210", "lr211"],
    },
    "lrc2": {
        "train": ["lrc201", "lrc202", "lrc203", "lrc204", "lrc205"],
        "val": ["lrc206"],
        "test": ["lrc207", "lrc208"],
    },
}


def _concat(classes: Iterable[str], split: str) -> list[str]:
    files: list[str] = []
    for cls in classes:
        files.extend(SPLITS[cls][split])
    return files


MODEL_CONFIGS: dict[str, dict[str, list[str]]] = {
    "lc1": SPLITS["lc1"],
    "lr1": SPLITS["lr1"],
    "lrc1": SPLITS["lrc1"],
    "lc2": SPLITS["lc2"],
    "lr2": SPLITS["lr2"],
    "lrc2": SPLITS["lrc2"],
    "mixed_c1": {
        "train": _concat(["lc1", "lr1", "lrc1"], "train"),
        "val": _concat(["lc1", "lr1", "lrc1"], "val"),
        "test": _concat(["lc1", "lr1", "lrc1"], "test"),
    },
    "mixed_c2": {
        "train": _concat(["lc2", "lr2", "lrc2"], "train"),
        "val": _concat(["lc2", "lr2", "lrc2"], "val"),
        "test": _concat(["lc2", "lr2", "lrc2"], "test"),
    },
    "mixed_all": {
        "train": _concat(["lc1", "lr1", "lrc1", "lc2", "lr2", "lrc2"], "train"),
        "val": _concat(["lc1", "lr1", "lrc1", "lc2", "lr2", "lrc2"], "val"),
        "test": _concat(["lc1", "lr1", "lrc1", "lc2", "lr2", "lrc2"], "test"),
    },
    # 2026-07-10: first NYC smoke test - only one instance exists so far (RollingHorizon
    # run nyc_morning500_mc3, labels from strict temporal-overlap pairs, see
    # rtv_solver/pipeline/build_nyc_manifest.py), so train/val/test all point at the
    # same file. This is not a generalization test, just a feasibility check that the
    # pipeline trains at all on NYC-derived data. Run with
    # --manifest_dir solutions/nyc/manifests --models nyc_morning500_mc3.
    "nyc_morning500_mc3": {
        "train": ["nyc_morning500_mc3"],
        "val": ["nyc_morning500_mc3"],
        "test": ["nyc_morning500_mc3"],
    },
}

DEFAULT_POS_WEIGHTS = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_weight_name(pos_weight: float) -> str:
    # 7.5 -> 7p5; 1.0 -> 1
    if float(pos_weight).is_integer():
        return str(int(pos_weight))
    return str(pos_weight).replace(".", "p")


def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


@dataclass
class DatasetInfo:
    name: str
    split: str
    num_nodes: int
    num_edges: int
    positives: int
    negatives: int


@dataclass
class MetricLog:
    run_name: str
    base_model: str
    pos_weight: float
    split: str
    instance: str
    epoch: int
    threshold: float
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    f3: float
    f4: float
    tp: int
    fp: int
    fn: int
    tn: int
    mean_pos: float
    mean_neg: float


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_graph_dataset(payload_path: Path, config: Config, split: str) -> dict | None:
    if not payload_path.exists():
        logging.warning("Missing file, skipping: %s", payload_path)
        return None

    payload = PayloadParser.load_input_data(payload_path)

    NetworkHandler.init_from_payload(
        payload=payload,
        server_url=config.SERVER_URL,
    )

    request_handler = RequestHandler(payload[PayloadKeys.REQUESTS], config)
    requests = request_handler.get_all_requests()

    G = RequestGraphFullBuilder.build(requests)
    G = RequestGraphFeatureBuilder.add_features(G)

    node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

    node_df = pd.DataFrame(node_matrix, columns=RequestGraphFeatureBuilder.NODE_FEATURES)
    edge_df = pd.DataFrame(edge_matrix, columns=RequestGraphFeatureBuilder.EDGE_FEATURES)

    labels_np = RequestGraphLabelBuilder.build_expert_labels_from_solution_payload(
        edge_index=edge_index,
        node_ids=node_ids,
        solution_payload=payload,
    )

    # Per-instance standardization. Keep consistent with current inference pipeline.
    node_matrix_norm = ((node_df - node_df.mean()) / (node_df.std() + 1e-8)).to_numpy()
    edge_matrix_norm = ((edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)).to_numpy()

    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_index_reindexed = [[id_to_idx[u], id_to_idx[v]] for u, v in edge_index]

    positives = int(labels_np.sum())
    negatives = int(len(labels_np) - positives)

    return {
        "name": payload_path.stem,
        "split": split,
        "x": torch.tensor(node_matrix_norm, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index_reindexed, dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(edge_matrix_norm, dtype=torch.float32),
        "labels": torch.tensor(labels_np, dtype=torch.float32),
        "num_nodes": len(node_ids),
        "num_edges": len(edge_index),
        "positives": positives,
        "negatives": negatives,
    }


def load_datasets(file_names: list[str], manifest_dir: Path, config: Config, split: str) -> list[dict]:
    datasets: list[dict] = []
    for name in file_names:
        ds = build_graph_dataset(manifest_dir / f"{name}.json", config, split=split)
        if ds is not None:
            datasets.append(ds)
            ratio = ds["positives"] / max(ds["num_edges"], 1)
            logging.info(
                "  %-5s %-8s nodes=%4d edges=%6d pos=%5d neg=%6d ratio=%.4f",
                split,
                ds["name"],
                ds["num_nodes"],
                ds["num_edges"],
                ds["positives"],
                ds["negatives"],
                ratio,
            )
    return datasets


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def balanced_indices(labels: torch.Tensor) -> torch.Tensor:
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]

    if len(pos_idx) == 0:
        return neg_idx[: min(10, len(neg_idx))]
    if len(neg_idx) == 0:
        return pos_idx

    n = min(len(pos_idx), len(neg_idx))
    sampled_pos = pos_idx[torch.randperm(len(pos_idx))[:n]]
    sampled_neg = neg_idx[torch.randperm(len(neg_idx))[:n]]
    idx = torch.cat([sampled_pos, sampled_neg])
    return idx[torch.randperm(len(idx))]


@torch.no_grad()
def evaluate(model: nn.Module, dataset: dict, threshold: float) -> dict:
    model.eval()
    logits = model(
        x=dataset["x"],
        edge_index=dataset["edge_index"],
        edge_attr=dataset["edge_attr"],
    )
    scores = torch.sigmoid(logits)
    labels = dataset["labels"]
    preds = scores > threshold

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": float((preds.float() == labels).float().mean().item()),
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1),
        "f2": f_beta(precision, recall, 2),
        "f3": f_beta(precision, recall, 3),
        "f4": f_beta(precision, recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "mean_pos": float(scores[labels == 1].mean().item()) if (labels == 1).any() else 0.0,
        "mean_neg": float(scores[labels == 0].mean().item()) if (labels == 0).any() else 0.0,
    }


def average_metric(model: nn.Module, datasets: list[dict], key: str, threshold: float) -> float:
    if not datasets:
        return 0.0
    return sum(evaluate(model, ds, threshold)[key] for ds in datasets) / len(datasets)


def append_logs(
    logs: list[MetricLog],
    model: nn.Module,
    datasets: list[dict],
    run_name: str,
    base_model: str,
    pos_weight: float,
    epoch: int,
    loss: float,
    thresholds: list[float],
) -> None:
    for ds in datasets:
        for threshold in thresholds:
            m = evaluate(model, ds, threshold)
            logs.append(
                MetricLog(
                    run_name=run_name,
                    base_model=base_model,
                    pos_weight=pos_weight,
                    split=ds["split"],
                    instance=ds["name"],
                    epoch=epoch,
                    threshold=threshold,
                    loss=loss,
                    **m,
                )
            )


# ---------------------------------------------------------------------------
# One training run



def train_one_run(
    base_model: str,
    pos_weight: float,
    model_split: dict[str, list[str]],
    manifest_dir: Path,
    output_dir: Path,
    config: Config,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    message_passing_steps: int,
    thresholds: list[float],
    patience: int,
    seed: int,
    gnn_version: str = "v2",
) -> list[MetricLog]:
    set_seed(seed)

    pw_name = safe_weight_name(pos_weight)
    # 2026-07-07: suffix now reflects the actual architecture (gnn_version), not the
    # training-script version - previously this was hardcoded to "_v2" regardless of
    # which model class was trained, which is exactly the naming confusion that led
    # to the v1/v2 mismatch bug (see request_graph_pruner.py).
    run_name = f"rgnn_{base_model}_pw{pw_name}_{gnn_version}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = run_dir / f"{run_name}_final.pt"
    best_model_path = run_dir / f"{run_name}_best_val_f3.pt"
    metadata_path = run_dir / "metadata.json"

    logging.info("\n%s", "=" * 80)
    logging.info("Run: %s", run_name)
    logging.info("Output: %s", run_dir)
    logging.info("Train files: %s", model_split["train"])
    logging.info("Val files:   %s", model_split["val"])
    logging.info("Test files:  %s", model_split["test"])

    train_sets = load_datasets(model_split["train"], manifest_dir, config, "train")
    val_sets = load_datasets(model_split["val"], manifest_dir, config, "val")
    test_sets = load_datasets(model_split["test"], manifest_dir, config, "test")

    if not train_sets:
        logging.error("No training datasets loaded for %s", run_name)
        return []

    first = train_sets[0]
    # 2026-07-07: pick the architecture explicitly by gnn_version so the same sweep
    # (splits, pos_weights, thresholds, depth) can be trained for v1 and v2 and
    # compared head-to-head. Depth is passed through under each class's own
    # constructor name (message_passing_steps vs. num_layers) - same CLI value,
    # so runs stay comparable across architectures.
    if gnn_version == "v1":
        model = RequestGraphEdgeGNN(
            node_feature_dim=first["x"].shape[1],
            edge_feature_dim=first["edge_attr"].shape[1],
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
        )
    else:
        model = RequestGraphEdgeGNNv2(
            node_feature_dim=first["x"].shape[1],
            edge_feature_dim=first["edge_attr"].shape[1],
            hidden_dim=hidden_dim,
            num_layers=message_passing_steps,
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(3, patience // 3),
    )

    logs: list[MetricLog] = []
    best_val_f3 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    t0 = time.time()

    metadata = {
        "run_name": run_name,
        "gnn_version": gnn_version,
        "base_model": base_model,
        "pos_weight": pos_weight,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "message_passing_steps": message_passing_steps,
        "thresholds": thresholds,
        "splits": model_split,
        "node_features": RequestGraphFeatureBuilder.NODE_FEATURES,
        "edge_features": RequestGraphFeatureBuilder.EDGE_FEATURES,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for ds in train_sets:
            idx = balanced_indices(ds["labels"])
            optimizer.zero_grad()
            logits = model(ds["x"], ds["edge_index"], ds["edge_attr"])
            loss = criterion(logits[idx], ds["labels"][idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / len(train_sets)

        # Select checkpoint by average validation F3 at threshold 0.5.
        # We log other thresholds, but checkpoint selection stays stable and simple.
        selection_threshold = 0.5
        val_f3 = average_metric(model, val_sets, "f3", selection_threshold) if val_sets else 0.0
        val_recall = average_metric(model, val_sets, "recall", selection_threshold) if val_sets else 0.0
        val_precision = average_metric(model, val_sets, "precision", selection_threshold) if val_sets else 0.0
        scheduler.step(val_f3)

        improved = val_f3 > best_val_f3 + 1e-8
        if improved:
            best_val_f3 = val_f3
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == epochs - 1 or improved:
            append_logs(
                logs=logs,
                model=model,
                datasets=train_sets + val_sets + test_sets,
                run_name=run_name,
                base_model=base_model,
                pos_weight=pos_weight,
                epoch=epoch,
                loss=avg_loss,
                thresholds=thresholds,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                "Epoch %03d/%03d | loss=%.4f | val_prec=%.3f | val_rec=%.3f | val_f3=%.3f | best_f3=%.3f @%d | lr=%.5f | %.0fs",
                epoch,
                epochs - 1,
                avg_loss,
                val_precision,
                val_recall,
                val_f3,
                best_val_f3,
                best_epoch,
                current_lr,
                time.time() - t0,
            )

        if patience > 0 and epochs_without_improvement >= patience:
            logging.info("Early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
            break

    torch.save(model.state_dict(), final_model_path)

    # Evaluate best checkpoint once more and log final best metrics at epoch=-1.
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        append_logs(
            logs=logs,
            model=model,
            datasets=train_sets + val_sets + test_sets,
            run_name=run_name,
            base_model=base_model,
            pos_weight=pos_weight,
            epoch=-1,
            loss=float("nan"),
            thresholds=thresholds,
        )

    logging.info("Saved final: %s", final_model_path)
    logging.info("Saved best:  %s", best_model_path)

    # Per-run log.
    if logs:
        pd.DataFrame([asdict(x) for x in logs]).to_csv(run_dir / "metrics_log.csv", index=False)

    return logs



# Summary tables


def write_summaries(all_logs: list[MetricLog], output_dir: Path) -> None:
    if not all_logs:
        return

    df = pd.DataFrame([asdict(x) for x in all_logs])
    all_log_path = output_dir / "training_metrics_all.csv"
    df.to_csv(all_log_path, index=False)

    # Best checkpoint rows only; threshold-specific validation summary.
    best_rows = df[df["epoch"] == -1].copy()
    if best_rows.empty:
        best_rows = df[df["epoch"] == df["epoch"].max()].copy()

    for split_name in ["train", "val", "test"]:
        split_df = best_rows[best_rows["split"] == split_name]
        if split_df.empty:
            continue
        summary = (
            split_df.groupby(["run_name", "base_model", "pos_weight", "threshold"])[
                ["accuracy", "precision", "recall", "f1", "f2", "f3", "f4", "tp", "fp", "fn", "tn", "mean_pos", "mean_neg"]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["f3", "recall"], ascending=False)
        )
        summary.to_csv(output_dir / f"summary_{split_name}_best_checkpoint.csv", index=False)

    # One row per base model: best validation configuration by F3.
    val_summary_path = output_dir / "summary_val_best_checkpoint.csv"
    if val_summary_path.exists():
        val_summary = pd.read_csv(val_summary_path)
        idx = val_summary.groupby("base_model")["f3"].idxmax()
        best_by_model = val_summary.loc[idx].sort_values("base_model")
        best_by_model.to_csv(output_dir / "best_config_by_model_val_f3.csv", index=False)

    logging.info("Saved full log: %s", all_log_path)
    logging.info("Saved summaries under: %s", output_dir)


# Main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train request-graph GNN models with clean splits and loss sweep.")
    parser.add_argument("--manifest_dir", default="solutions/li_lim/manifests")
    # 2026-07-07: default is None so it can depend on --gnn_version (see main()) -
    # v1 and v2 runs land in separate folders (outputs/models_v2 vs.
    # outputs/models_v2_gnnv2) so a full sweep of both never overwrites the other.
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--gnn_version",
        choices=["v1", "v2"],
        default="v2",
        help="Which architecture to train: v1 = RequestGraphEdgeGNN (shared weights, "
        "edge embeddings frozen after encoder), v2 = RequestGraphEdgeGNNv2 (per-layer "
        "weights, edge updates + residual + LayerNorm every layer).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--pos_weights", nargs="+", type=float, default=DEFAULT_POS_WEIGHTS)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--message_passing_steps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(
            "outputs/models_v2" if args.gnn_version == "v1" else "outputs/models_v2_gnnv2"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir)

    logging.info("%s", "=" * 80)
    logging.info("REQUEST-GRAPH GNN TRAINING V2")
    logging.info("%s", "=" * 80)
    logging.info("GNN architecture: %s", args.gnn_version)
    logging.info("Output dir: %s", output_dir)
    logging.info("Models: %s", args.models)
    logging.info("pos_weights: %s", args.pos_weights)
    logging.info("thresholds: %s", args.thresholds)
    logging.info("epochs=%d lr=%.5f hidden=%d mp_steps=%d patience=%d seed=%d", args.epochs, args.learning_rate, args.hidden_dim, args.message_passing_steps, args.patience, args.seed)

    config = Config()
    all_logs: list[MetricLog] = []
    total_start = time.time()

    for base_model in args.models:
        split = MODEL_CONFIGS[base_model]
        for pos_weight in args.pos_weights:
            logs = train_one_run(
                base_model=base_model,
                pos_weight=pos_weight,
                model_split=split,
                manifest_dir=manifest_dir,
                output_dir=output_dir,
                config=config,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                message_passing_steps=args.message_passing_steps,
                thresholds=args.thresholds,
                patience=args.patience,
                seed=args.seed,
                gnn_version=args.gnn_version,
            )
            all_logs.extend(logs)
            # Persist after every run so partial results are safe.
            write_summaries(all_logs, output_dir)

    elapsed_min = (time.time() - total_start) / 60.0
    logging.info("DONE in %.1f minutes", elapsed_min)
    print("\nDONE")
    print(f"Models and logs saved under: {output_dir}")
    print("Best configs by validation F3:")
    best_path = output_dir / "best_config_by_model_val_f3.csv"
    if best_path.exists():
        print(best_path)


if __name__ == "__main__":
    main()
