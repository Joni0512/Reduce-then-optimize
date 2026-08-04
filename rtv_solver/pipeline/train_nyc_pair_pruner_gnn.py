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
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder


# ------------------------------------------------------------
# Settings
# 2026-08-02: first NYC pair-pruner test run - v1 GNN, natural
# (unbalanced) class distribution, POS_WEIGHT=1 (i.e. no imbalance
# correction at all), threshold fixed at 0.5. Train on Day A only,
# evaluate on Day B (Val) only - Day C (Test) intentionally held out
# until a final config is chosen, same convention used throughout
# the SIL granularity experiments this session.
# ------------------------------------------------------------

TRAIN_PATH = Path("solutions/nyc/manifests/nyc_real1000_20160112_0614_train_v50_expert.json")
VAL_PATH = Path("solutions/nyc/manifests/nyc_real1000_20160113_0614_val_v50_expert.json")

EPOCHS = 100
LEARNING_RATE = 1e-2
HIDDEN_DIM = 64
MESSAGE_PASSING_STEPS = 1  # 2026-08-04: testing 1 layer vs. the 2- and 3-layer results
POS_WEIGHT = 1.0  # 2026-08-04: 1-layer test
BALANCED = True
THRESHOLD = 0.5
F_BETA = 3.0  # matches the F3 convention already used for pruner checkpoint selection elsewhere (see REQUEST_GRAPH_MODEL_PATH in config.py)

MODEL_PATH = Path("outputs/models_nyc/pair_pruner_v1_1l_pw1_balanced/pair_pruner_v1_1l_pw1_balanced_best_val_f3.pt")


def build_graph_dataset(payload_path: Path):
    """
    Loads one NYC expert-manifest payload and converts it into tensors for
    the GNN. Identical to train_request_graph_gnn_multi.py's
    build_graph_dataset, except geographic=True is passed to the feature
    builder (2026-08-02: NYC's lat/lon are real WGS84 degrees, not Li&Lim's
    flat grid coordinates - see RequestGraphFeatureBuilder.add_features).
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
    G = RequestGraphFeatureBuilder.add_features(G, geographic=True)

    node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

    node_df = pd.DataFrame(
        node_matrix,
        columns=RequestGraphFeatureBuilder.NODE_FEATURES,
    )

    edge_df = pd.DataFrame(
        edge_matrix,
        columns=RequestGraphFeatureBuilder.EDGE_FEATURES,
    )

    # Label = 1 if two requests are ever served by the same vehicle in the
    # RHO expert baseline (build_expert_labels_from_solution_payload reads
    # this straight from `payload["driver_runs"]`, which is already present
    # in our NYC _expert.json files - same self-referential structure as
    # Li&Lim's manifests).
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


# ------------------------------------------------------------
# Balanced-sampling helper (copied unchanged from train_request_graph_gnn_multi.py)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Evaluation: accuracy/mean-score diagnostics plus precision/recall/F-beta
# 2026-08-02: precision/recall/F3 added - accuracy alone is meaningless at
# NYC's ~1% positive rate (always-negative baseline already scores ~99%).
# ------------------------------------------------------------

def evaluate_dataset(model, dataset, threshold: float = THRESHOLD, beta: float = F_BETA):
    model.eval()

    with torch.no_grad():
        logits = model(
            x=dataset["x"],
            edge_index=dataset["edge_index"],
            edge_attr=dataset["edge_attr"],
        )

        scores = torch.sigmoid(logits)
        labels = dataset["labels"]

        predictions = (scores > threshold).float()

        accuracy = (predictions == labels).float().mean().item()
        mean_pos = scores[labels == 1].mean().item()
        mean_neg = scores[labels == 0].mean().item()
        kept_edges = int(predictions.sum().item())

        tp = float(((predictions == 1) & (labels == 1)).sum().item())
        fp = float(((predictions == 1) & (labels == 0)).sum().item())
        fn = float(((predictions == 0) & (labels == 1)).sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = beta ** 2
        f_beta = (
            (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return {
        "accuracy": accuracy,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "kept_edges": kept_edges,
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
    }


def main():
    print("Loading NYC train/val instances...")

    train_dataset = build_graph_dataset(TRAIN_PATH)
    val_dataset = build_graph_dataset(VAL_PATH)

    for label, dataset in [("TRAIN", train_dataset), ("VAL", val_dataset)]:
        positives = int(dataset["labels"].sum().item())
        negatives = int(dataset["labels"].numel() - positives)
        print(
            f"{label} {dataset['name']} | "
            f"nodes={dataset['num_nodes']} | "
            f"edges={dataset['num_edges']} | "
            f"positive={positives} | "
            f"negative={negatives} | "
            f"positive_ratio={positives / dataset['num_edges']:.4f}"
        )

    model = RequestGraphEdgeGNN(
        node_feature_dim=train_dataset["x"].shape[1],
        edge_feature_dim=train_dataset["edge_attr"].shape[1],
        hidden_dim=HIDDEN_DIM,
        message_passing_steps=MESSAGE_PASSING_STEPS,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT])
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nStart NYC pair-pruner training (v1 architecture, single train day)...")

    best_val_f_beta = -1.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad()

        logits = model(
            x=train_dataset["x"],
            edge_index=train_dataset["edge_index"],
            edge_attr=train_dataset["edge_attr"],
        )

        if BALANCED:
            # 2026-08-02: fresh 50:50 sample each epoch, loss computed only
            # on this subset - isolates the balancing effect from POS_WEIGHT,
            # which stays at 1 this run.
            train_idx = balanced_train_indices(train_dataset["labels"])
            loss = criterion(logits[train_idx], train_dataset["labels"][train_idx])
        else:
            loss = criterion(logits, train_dataset["labels"])

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            train_diag = evaluate_dataset(model, train_dataset)
            val_diag = evaluate_dataset(model, val_dataset)

            print(
                f"Epoch {epoch:03d} | loss: {loss.item():.4f} | "
                f"train P/R/F3: {train_diag['precision']:.3f}/{train_diag['recall']:.3f}/{train_diag['f_beta']:.3f} | "
                f"val P/R/F3: {val_diag['precision']:.3f}/{val_diag['recall']:.3f}/{val_diag['f_beta']:.3f} | "
                f"val acc: {val_diag['accuracy']:.4f} | val kept_edges: {val_diag['kept_edges']}"
            )

            if val_diag["f_beta"] > best_val_f_beta:
                best_val_f_beta = val_diag["f_beta"]
                best_epoch = epoch
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)

    print(
        f"\nBest val F{F_BETA:.0f} = {best_val_f_beta:.4f} at epoch {best_epoch} "
        f"-> {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
