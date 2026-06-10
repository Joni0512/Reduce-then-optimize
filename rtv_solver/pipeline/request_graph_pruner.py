from pathlib import Path

import pandas as pd
import torch

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN


class RequestGraphPruner:
    """
    Baut aus aktiven Requests erlaubte Request-Paare für die Trip-Generation.

    Idee:
        Requests
        ↓
        vollständiger Request-Graph
        ↓
        GNN Scores
        ↓
        Threshold
        ↓
        allowed_pairs

    allowed_pairs wird später im TripHandler genutzt:
        Trip [A, B, C] ist nur erlaubt,
        wenn (A,B), (A,C), (B,C) in allowed_pairs sind.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps
        self.device = torch.device(device)

        self.model = None

    def build_allowed_pairs(self, requests):
        """
        Main function.

        Input:
            requests:
                Liste von Request-Objekten aus dem aktuellen Solver-Zeitfenster.

        Output:
            allowed_pairs:
                set[(request_id_a, request_id_b)]

            stats:
                dict mit Pruning-Statistiken
        """

        if len(requests) <= 1:
            return set(), {
                "num_requests": len(requests),
                "num_edges_total": 0,
                "num_edges_kept": 0,
                "edge_reduction": 0.0,
                "threshold": self.threshold,
            }

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

        node_matrix_norm = (
            (node_df - node_df.mean()) / (node_df.std() + 1e-8)
        ).to_numpy()

        edge_matrix_norm = (
            (edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)
        ).to_numpy()

        x = torch.tensor(
            node_matrix_norm,
            dtype=torch.float32,
            device=self.device,
        )

        edge_attr = torch.tensor(
            edge_matrix_norm,
            dtype=torch.float32,
            device=self.device,
        )

        id_to_idx = {
            node_id: idx
            for idx, node_id in enumerate(node_ids)
        }

        edge_index_reindexed = [
            [id_to_idx[u], id_to_idx[v]]
            for u, v in edge_index
        ]

        edge_index_t = torch.tensor(
            edge_index_reindexed,
            dtype=torch.long,
            device=self.device,
        ).t().contiguous()

        self._load_model_if_needed(
            node_feature_dim=x.shape[1],
            edge_feature_dim=edge_attr.shape[1],
        )

        self.model.eval()

        with torch.no_grad():
            logits = self.model(
                x=x,
                edge_index=edge_index_t,
                edge_attr=edge_attr,
            )
            scores = torch.sigmoid(logits).cpu().numpy()

        allowed_pairs = set()

        for (u, v), score in zip(edge_index, scores):
            if score >= self.threshold:
                allowed_pairs.add(tuple(sorted((int(u), int(v)))))

        num_edges_total = len(edge_index)
        num_edges_kept = len(allowed_pairs)

        stats = {
            "num_requests": len(requests),
            "num_edges_total": num_edges_total,
            "num_edges_kept": num_edges_kept,
            "edge_reduction": (
                1.0 - num_edges_kept / num_edges_total
                if num_edges_total > 0
                else 0.0
            ),
            "threshold": self.threshold,
        }

        return allowed_pairs, stats

    def _load_model_if_needed(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
    ):
        if self.model is not None:
            return

        self.model = RequestGraphEdgeGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=self.hidden_dim,
            message_passing_steps=self.message_passing_steps,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                self.model_path,
                map_location=self.device,
            )
        )


def trip_allowed_by_request_pairs(
    request_ids,
    allowed_pairs,
) -> bool:
    """
    Prüft, ob ein Trip durch den geprunten Request-Graph erlaubt ist.

    Single-request trips bleiben immer erlaubt.

    Für Trips mit mehreren Requests:
        Alle Request-Paare müssen in allowed_pairs vorkommen.
    """

    if allowed_pairs is None:
        return True

    if len(request_ids) <= 1:
        return True

    for i in range(len(request_ids)):
        for j in range(i + 1, len(request_ids)):
            pair = tuple(sorted((int(request_ids[i]), int(request_ids[j]))))

            if pair not in allowed_pairs:
                return False

    return True