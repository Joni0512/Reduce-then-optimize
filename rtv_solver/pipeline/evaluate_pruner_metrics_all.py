"""
Evaluates the GNN pruner classifier - both architectures - against expert labels.

For every test instance, every pruning threshold (0.1-0.5), and both GNN
checkpoints (v1 and v2), computes classification metrics (accuracy, precision,
recall, F1, F2, F3) for the "keep this request-pair edge" decision: does the
pruner keep edges that the expert solution actually uses (true positives), and
does it wrongly drop edges the expert solution needs (false negatives)?

This is independent of solver pipeline (COAML vs. RH/offline) - pruning happens
before either solver runs and depends only on (instance, GNN checkpoint,
threshold), so metrics are computed once per (instance, gnn_version, threshold),
not duplicated per pipeline.

Usage:
    python -m rtv_solver.pipeline.evaluate_pruner_metrics_all
"""

from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_gnn_v2 import RequestGraphEdgeGNNv2
from rtv_solver.pipeline.request_graph_pruner import _detect_gnn_version


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "solutions" / "li_lim" / "manifests"

# Same 12 instances used throughout the card2_firstcheck solver sweep.
TEST_FILES = [
    "lc108.json", "lc109.json", "lc207.json", "lc208.json",
    "lr111.json", "lr112.json", "lr210.json", "lr211.json",
    "lrc107.json", "lrc108.json", "lrc207.json", "lrc208.json",
]

# Same checkpoints used as MODEL_V1 / MODEL_V2 across the card2_firstcheck run commands.
GNN_CHECKPOINTS = {
    "gnn_v1": ROOT / "outputs" / "models_v2" / "rgnn_mixed_c2_pw2_v2" / "rgnn_mixed_c2_pw2_v2_best_val_f3.pt",
    "gnn_v2": ROOT / "outputs" / "models_v2_gnnv2" / "rgnn_mixed_c2_pw10_v2" / "rgnn_mixed_c2_pw10_v2_best_val_f3.pt",
}
HIDDEN_DIM = 64

# Same thresholds as the solver sweep (0.0 = "keep everything" is not a meaningful
# classifier operating point, so we start at 0.1).
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Match the location of the rest of the card2_firstcheck sweep analysis so results
# sit next to the runtime/service-rate CSVs and plots from parse_card2_firstcheck.py.
OUTPUT_DIR = ROOT / "outputs" / "outputs" / "new_tests" / "card2_firstcheck" / "analysis" / "pruner_metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"


def instance_class(instance: str) -> str:
    if instance.startswith("lrc"):
        return "LRC"
    if instance.startswith("lr"):
        return "LR"
    if instance.startswith("lc"):
        return "LC"
    return "UNKNOWN"


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def fbeta(precision: float, recall: float, beta: float) -> float:
    denom = (beta ** 2) * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / denom


def build_graph_dataset(payload_path: Path):
    config = Config()
    payload = PayloadParser.load_input_data(payload_path)

    NetworkHandler.init_from_payload(
        payload=payload,
        server_url=config.SERVER_URL,
    )

    request_handler = RequestHandler(
        payload[PayloadKeys.REQUESTS],
        config,
    )

    requests = request_handler.get_all_requests()

    G = RequestGraphFullBuilder.build(requests)
    G = RequestGraphFeatureBuilder.add_features(G)

    node_matrix, edge_index, edge_matrix, node_ids = (
        RequestGraphFeatureBuilder.to_numpy(G)
    )

    node_df = pd.DataFrame(node_matrix, columns=RequestGraphFeatureBuilder.NODE_FEATURES)
    edge_df = pd.DataFrame(edge_matrix, columns=RequestGraphFeatureBuilder.EDGE_FEATURES)

    labels_np = RequestGraphLabelBuilder.build_expert_labels_from_solution_payload(
        edge_index=edge_index,
        node_ids=node_ids,
        solution_payload=payload,
    )

    node_matrix_norm = ((node_df - node_df.mean()) / (node_df.std() + 1e-8)).to_numpy()
    edge_matrix_norm = ((edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)).to_numpy()

    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_index_reindexed = [[id_to_idx[u], id_to_idx[v]] for u, v in edge_index]

    return {
        "name": payload_path.stem,
        "x": torch.tensor(node_matrix_norm, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index_reindexed, dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(edge_matrix_norm, dtype=torch.float32),
        "labels": torch.tensor(labels_np, dtype=torch.float32),
        "num_nodes": len(node_ids),
        "num_edges": len(edge_index),
    }


def load_model(checkpoint_path: Path, node_feature_dim: int, edge_feature_dim: int):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    version = _detect_gnn_version(state_dict)

    if version == "v2":
        layer_indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("message_mlps.")
        }
        num_layers = max(layer_indices) + 1
        model = RequestGraphEdgeGNNv2(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=num_layers,
        )
    else:
        model = RequestGraphEdgeGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=HIDDEN_DIM,
            message_passing_steps=2,
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model, version


def evaluate_instance(model, gnn_version, dataset):
    with torch.no_grad():
        logits = model(
            x=dataset["x"],
            edge_index=dataset["edge_index"],
            edge_attr=dataset["edge_attr"],
        )
        scores = torch.sigmoid(logits)

    labels = dataset["labels"]

    total_edges = labels.numel()
    total_positive = int((labels == 1).sum().item())
    total_negative = int((labels == 0).sum().item())

    rows = []

    for threshold in THRESHOLDS:
        # ">=" matches RequestGraphPruner.build_allowed_pairs at runtime.
        keep_mask = scores >= threshold

        edges_kept = int(keep_mask.sum().item())
        tp = int(((labels == 1) & keep_mask).sum().item())
        fp = int(((labels == 0) & keep_mask).sum().item())
        fn = int(((labels == 1) & ~keep_mask).sum().item())
        tn = int(((labels == 0) & ~keep_mask).sum().item())

        recall = tp / total_positive if total_positive > 0 else 0.0
        precision = tp / edges_kept if edges_kept > 0 else 0.0
        accuracy = (tp + tn) / total_edges if total_edges > 0 else 0.0
        f1 = fbeta(precision, recall, 1.0)
        f2 = fbeta(precision, recall, 2.0)
        f3 = fbeta(precision, recall, 3.0)

        edge_reduction = 1.0 - (edges_kept / total_edges) if total_edges > 0 else 0.0

        rows.append({
            "gnn_version": gnn_version,
            "instance": dataset["name"],
            "class": instance_class(dataset["name"]),
            "threshold": threshold,
            "nodes": dataset["num_nodes"],
            "edges_total": total_edges,
            "positive_edges_total": total_positive,
            "negative_edges_total": total_negative,
            "edges_kept": edges_kept,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "edge_reduction": edge_reduction,
        })

    return rows


def make_plots(df: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f2", "f3", "edge_reduction"]
    colors = {"gnn_v1": "tab:blue", "gnn_v2": "tab:orange"}

    # v1 vs v2, mean +/- IQR/2 band, one figure per metric.
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for gnn_version, color in colors.items():
            sub = df[df["gnn_version"] == gnn_version]
            stats = sub.groupby("threshold")[metric].agg(["mean", iqr]).reset_index()
            ax.plot(stats["threshold"], stats["mean"], marker="o", color=color, label=gnn_version)
            ax.fill_between(
                stats["threshold"],
                stats["mean"] - stats["iqr"] / 2,
                stats["mean"] + stats["iqr"] / 2,
                alpha=0.15,
                color=color,
            )
        ax.set_xlabel("threshold")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. threshold, v1 vs v2 (mean +/- IQR/2)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{metric}_v1_vs_v2.png", dpi=150)
        plt.close(fig)

    # Per-instance detail, faceted by class, one figure per (metric, gnn_version).
    for gnn_version in colors:
        for metric in ["precision", "recall", "f2"]:
            sub_v = df[df["gnn_version"] == gnn_version]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
            for ax, cls in zip(axes, ["LC", "LR", "LRC"]):
                sub = sub_v[sub_v["class"] == cls]
                for instance, g in sub.groupby("instance"):
                    g = g.sort_values("threshold")
                    ax.plot(g["threshold"], g[metric], marker="o", label=instance)
                ax.set_title(cls)
                ax.set_xlabel("threshold")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)
            axes[0].set_ylabel(metric)
            fig.suptitle(f"{gnn_version}: {metric} vs. threshold, per instance")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{metric}_{gnn_version}_per_instance.png", dpi=150)
            plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        filename: build_graph_dataset(BASE_DIR / filename)
        for filename in TEST_FILES
    }

    first_dataset = next(iter(datasets.values()))

    all_rows = []

    for gnn_version, checkpoint_path in GNN_CHECKPOINTS.items():
        model, detected_version = load_model(
            checkpoint_path,
            node_feature_dim=first_dataset["x"].shape[1],
            edge_feature_dim=first_dataset["edge_attr"].shape[1],
        )
        assert f"gnn_{detected_version}" == gnn_version, (
            f"Checkpoint {checkpoint_path} was expected to be {gnn_version} "
            f"but state_dict keys indicate gnn_{detected_version}"
        )

        for filename, dataset in datasets.items():
            rows = evaluate_instance(model, gnn_version, dataset)
            all_rows.extend(rows)

            print(
                f"[{gnn_version}] Evaluated {filename} | "
                f"nodes={dataset['num_nodes']} | edges={dataset['num_edges']} | "
                f"positive={int(dataset['labels'].sum().item())}"
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_DIR / "pruner_metrics_long.csv", index=False)

    metrics = ["accuracy", "precision", "recall", "f1", "f2", "f3", "edge_reduction"]

    overall_summary = (
        df.groupby(["gnn_version", "threshold"])[metrics]
        .agg(["mean", "median", "std", iqr])
    )
    overall_summary.columns = ["_".join(c) for c in overall_summary.columns]
    overall_summary.reset_index().to_csv(
        OUTPUT_DIR / "pruner_metrics_summary_overall.csv", index=False
    )

    class_summary = (
        df.groupby(["gnn_version", "class", "threshold"])[metrics]
        .agg(["mean", "median", "std", iqr])
    )
    class_summary.columns = ["_".join(c) for c in class_summary.columns]
    class_summary.reset_index().to_csv(
        OUTPUT_DIR / "pruner_metrics_summary_by_class.csv", index=False
    )

    # Wide table: one row per (gnn_version, instance), columns per threshold for each metric.
    wide = df.pivot_table(
        index=["gnn_version", "class", "instance"],
        columns="threshold",
        values=metrics,
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_t{str(th).replace('.', '')}" for metric, th in wide.columns]
    wide.reset_index().to_csv(OUTPUT_DIR / "pruner_metrics_wide.csv", index=False)

    make_plots(df)

    print("\nOVERALL SUMMARY (mean), by gnn_version/threshold:")
    print(df.groupby(["gnn_version", "threshold"])[metrics].mean().round(4))

    print(f"\nSaved long CSV to: {OUTPUT_DIR / 'pruner_metrics_long.csv'}")
    print(f"Saved overall summary to: {OUTPUT_DIR / 'pruner_metrics_summary_overall.csv'}")
    print(f"Saved class summary to: {OUTPUT_DIR / 'pruner_metrics_summary_by_class.csv'}")
    print(f"Saved wide table to: {OUTPUT_DIR / 'pruner_metrics_wide.csv'}")
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
