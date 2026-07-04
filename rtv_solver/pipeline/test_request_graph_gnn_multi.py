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
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder


BASE_DIR = Path("solutions/li_lim/manifests")

TEST_FILES = [
    # LR Holdout
    "lr110.json",
    "lr111.json",
    "lr112.json",

    # Große LR Holdouts
    "lr209.json",
    "lr210.json",
    "lr211.json",
]
MODEL_PATH = Path("outputs/request_graph_gnn_mixed_big.pt")
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "pruning_metrics.csv"
OUTPUT_PNG = OUTPUT_DIR / "pruning_metrics_table.png"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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

    node_df = pd.DataFrame(
        node_matrix,
        columns=RequestGraphFeatureBuilder.NODE_FEATURES,
    )

    edge_df = pd.DataFrame(
        edge_matrix,
        columns=RequestGraphFeatureBuilder.EDGE_FEATURES,
    )

    labels_np = RequestGraphLabelBuilder.build_expert_labels_from_solution_payload(
        edge_index=edge_index,
        node_ids=node_ids,
        solution_payload=payload,
    )

    node_matrix_norm = (
        (node_df - node_df.mean()) / (node_df.std() + 1e-8)
    ).to_numpy()

    edge_matrix_norm = (
        (edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)
    ).to_numpy()

    id_to_idx = {
        node_id: idx
        for idx, node_id in enumerate(node_ids)
    }

    edge_index_reindexed = [
        [id_to_idx[u], id_to_idx[v]]
        for u, v in edge_index
    ]

    return {
        "name": payload_path.stem,
        "x": torch.tensor(node_matrix_norm, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index_reindexed, dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(edge_matrix_norm, dtype=torch.float32),
        "labels": torch.tensor(labels_np, dtype=torch.float32),
        "num_nodes": len(node_ids),
        "num_edges": len(edge_index),
    }


def evaluate_instance(model, dataset):
    model.eval()

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
        keep_mask = scores > threshold

        edges_kept = int(keep_mask.sum().item())
        positive_kept = int(((labels == 1) & keep_mask).sum().item())
        negative_kept = int(((labels == 0) & keep_mask).sum().item())

        positive_recall = (
            positive_kept / total_positive
            if total_positive > 0
            else 0.0
        )

        precision = (
            positive_kept / edges_kept
            if edges_kept > 0
            else 0.0
        )

        edge_reduction = 1.0 - (edges_kept / total_edges)

        rows.append({
            "instance": dataset["name"],
            "threshold": threshold,
            "nodes": dataset["num_nodes"],
            "edges_total": total_edges,
            "positive_edges_total": total_positive,
            "negative_edges_total": total_negative,
            "edges_kept": edges_kept,
            "positive_edges_kept": positive_kept,
            "negative_edges_kept": negative_kept,
            "positive_recall": positive_recall,
            "precision": precision,
            "edge_reduction": edge_reduction,
        })

    return rows


def save_table_as_png(df: pd.DataFrame, output_path: Path):
    """
    Saves a compact table as PNG for slides / OneNote.
    """

    display_df = df.copy()

    # nicer formatting
    for col in ["positive_recall", "precision", "edge_reduction"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")

    display_df["threshold"] = display_df["threshold"].map(lambda x: f"{x:.1f}")

    # keep table slide-friendly
    display_df = display_df[
        [
            "instance",
            "threshold",
            "nodes",
            "edges_total",
            "positive_edges_total",
            "edges_kept",
            "positive_edges_kept",
            "positive_recall",
            "precision",
            "edge_reduction",
        ]
    ]

    fig_height = max(4, 0.35 * len(display_df))
    fig_width = 16

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    first_dataset = build_graph_dataset(BASE_DIR / TEST_FILES[0])

    model = RequestGraphEdgeGNN(
        node_feature_dim=first_dataset["x"].shape[1],
        edge_feature_dim=first_dataset["edge_attr"].shape[1],
        hidden_dim=64,
        message_passing_steps=2,
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    all_rows = []

    for filename in TEST_FILES:
        dataset = build_graph_dataset(BASE_DIR / filename)
        rows = evaluate_instance(model, dataset)
        all_rows.extend(rows)

        print(
            f"Evaluated {filename} | "
            f"nodes={dataset['num_nodes']} | "
            f"edges={dataset['num_edges']} | "
            f"positive={int(dataset['labels'].sum().item())}"
        )

    results_df = pd.DataFrame(all_rows)

    results_df.to_csv(OUTPUT_CSV, index=False)
    save_table_as_png(results_df, OUTPUT_PNG)

    print("\nPRUNING METRICS")
    print(results_df.to_string(index=False))

    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    print(f"Saved PNG to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()