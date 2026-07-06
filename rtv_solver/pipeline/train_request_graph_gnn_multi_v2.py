from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_gnn_v2 import RequestGraphEdgeGNNv2
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

BASE_DIR = Path("solutions/li_lim/manifests")

TRAIN_FILES = [
    # LC 100er
    "lc101.json", "lc102.json", "lc103.json", "lc104.json", "lc105.json",
    "lc106.json", "lc107.json", "lc108.json", "lc109.json",

    # LC 200er
    "lc201.json", "lc202.json", "lc203.json", "lc204.json", "lc205.json",
    "lc206.json", "lc207.json", "lc208.json",

    # LR 100er
    "lr101.json", "lr102.json", "lr103.json", "lr104.json", "lr105.json",
    "lr106.json", "lr107.json", "lr108.json", "lr109.json",

    # LR 200er
    "lr201.json", "lr202.json", "lr203.json", "lr204.json", "lr205.json",
    "lr206.json", "lr207.json", "lr208.json",

    # LRC 100er
    "lrc101.json", "lrc102.json", "lrc103.json", "lrc104.json", "lrc105.json",
    "lrc106.json", "lrc107.json", "lrc108.json",

    # LRC 200er
    "lrc201.json", "lrc202.json", "lrc203.json", "lrc204.json", "lrc205.json",
    "lrc206.json", "lrc207.json", "lrc208.json",
]


EPOCHS = 100
LEARNING_RATE = 1e-2
HIDDEN_DIM = 64
NUM_LAYERS = 4          # v2: distinct weights per layer, so this is a real depth increase
DROPOUT = 0.1
POS_WEIGHT = 5.0

MODEL_PATH = Path("outputs/models_v2/rgnn_v2_arch_pw5/rgnn_v2_arch_pw5.pt")


def build_graph_dataset(payload_path: Path):
    """
    Loads one Li-Lim manifest payload and converts it into tensors for the GNN.
    Identical feature/label pipeline to train_request_graph_gnn_multi.py so the
    only difference vs. that script is the model architecture (v1 vs v2).
    """

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

    node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

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


def balanced_train_indices(labels: torch.Tensor) -> torch.Tensor:
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]

    if len(pos_idx) == 0:
        raise ValueError("No positive labels found.")

    sampled_neg_idx = neg_idx[
        torch.randperm(len(neg_idx))[:len(pos_idx)]
    ]

    train_idx = torch.cat([pos_idx, sampled_neg_idx])
    train_idx = train_idx[torch.randperm(len(train_idx))]

    return train_idx


def evaluate_dataset(model, dataset):
    model.eval()

    with torch.no_grad():
        logits = model(
            x=dataset["x"],
            edge_index=dataset["edge_index"],
            edge_attr=dataset["edge_attr"],
        )

        scores = torch.sigmoid(logits)
        labels = dataset["labels"]

        predictions = scores > 0.5

        accuracy = (predictions.float() == labels).float().mean().item()

        mean_pos = scores[labels == 1].mean().item()
        mean_neg = scores[labels == 0].mean().item()

        kept_edges = int(predictions.sum().item())

    return {
        "accuracy": accuracy,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "kept_edges": kept_edges,
    }


def main():
    print("Loading training instances...")

    train_datasets = []

    for filename in TRAIN_FILES:
        path = BASE_DIR / filename
        dataset = build_graph_dataset(path)
        train_datasets.append(dataset)

        positives = int(dataset["labels"].sum().item())
        negatives = int(dataset["labels"].numel() - positives)

        print(
            f"TRAIN {filename} | "
            f"nodes={dataset['num_nodes']} | "
            f"edges={dataset['num_edges']} | "
            f"positive={positives} | "
            f"negative={negatives} | "
            f"positive_ratio={positives / dataset['num_edges']:.4f}"
        )

    first_dataset = train_datasets[0]

    model = RequestGraphEdgeGNNv2(
        node_feature_dim=first_dataset["x"].shape[1],
        edge_feature_dim=first_dataset["edge_attr"].shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT])
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nStart multi-instance training (v2 architecture)...")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for dataset in train_datasets:
            labels = dataset["labels"]
            train_idx = balanced_train_indices(labels)

            optimizer.zero_grad()

            logits = model(
                x=dataset["x"],
                edge_index=dataset["edge_index"],
                edge_attr=dataset["edge_attr"],
            )

            loss = criterion(
                logits[train_idx],
                labels[train_idx],
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            avg_loss = total_loss / len(train_datasets)

            diagnostics = [
                evaluate_dataset(model, dataset)
                for dataset in train_datasets
            ]

            avg_accuracy = sum(d["accuracy"] for d in diagnostics) / len(diagnostics)
            avg_mean_pos = sum(d["mean_pos"] for d in diagnostics) / len(diagnostics)
            avg_mean_neg = sum(d["mean_neg"] for d in diagnostics) / len(diagnostics)

            print(
                f"Epoch {epoch:03d} | "
                f"Avg loss: {avg_loss:.4f} | "
                f"Avg acc: {avg_accuracy:.4f} | "
                f"Avg mean pos: {avg_mean_pos:.4f} | "
                f"Avg mean neg: {avg_mean_neg:.4f}"
            )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
