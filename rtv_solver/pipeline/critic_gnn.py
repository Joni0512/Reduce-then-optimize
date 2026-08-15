from __future__ import annotations

import torch
from torch import nn

from rtv_solver.pipeline.candidate_scoring_gnn import (
    GCNMeanLayer,
    GraphSAGEMeanLayer,
    GraphSAGEPoolLayer,
)
from rtv_solver.pipeline.match_graph_features import MatchGraphFeatureBuilder


class CriticGNN(nn.Module):
    """
    Critic network for SRL Phase 2 - one Q(s,a) value per MatchGraph
    (see match_solution_graph.py), built up step by step.

    Pipeline: two encoders -> message passing -> pooling -> Q-head.
    Pools requests and vehicles separately (mean each, then concat) instead
    of one pool over all nodes together, so the graph size (which varies a
    lot between iterations) does not by itself change the pooled scale, and
    request- vs. vehicle-side information stays distinguishable going into
    the head. 2026-08-12: no attention, no target network, no twin critic
    yet - keep V1 as simple as possible, add those later if needed.
    """

    def __init__(
        self,
        request_feature_dim: int = MatchGraphFeatureBuilder.REQUEST_FEATURE_SIZE,
        vehicle_feature_dim: int = MatchGraphFeatureBuilder.VEHICLE_FEATURE_SIZE,
        hidden_dim: int = 64,
        *,
        num_message_passing_layers: int = 2,
        aggregator: str = "gcn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if aggregator not in ("gcn", "mean", "pool"):
            raise ValueError(f"aggregator must be 'gcn', 'mean', or 'pool', got {aggregator!r}.")
        if num_message_passing_layers < 1:
            raise ValueError("num_message_passing_layers must be at least one.")

        self.request_feature_dim = request_feature_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.hidden_dim = hidden_dim

        # request and vehicle features have different meaning and length, so
        # each node type gets its own encoder - both map into the same
        # hidden_dim, so the two types can be mixed together afterwards
        self.request_encoder = nn.Sequential(
            nn.Linear(request_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # lets nodes pick up information from their graph neighbours - same
        # layer classes the actor GNN already uses, see candidate_scoring_gnn.py
        layer_classes = {
            "gcn": GCNMeanLayer,
            "mean": GraphSAGEMeanLayer,
            "pool": GraphSAGEPoolLayer,
        }
        layer_cls = layer_classes[aggregator]
        self.message_passing_layers = nn.ModuleList(
            layer_cls(hidden_dim, dropout=dropout) for _ in range(num_message_passing_layers)
        )

        # pooled vector is 2*hidden_dim wide (request mean concatenated with
        # vehicle mean), squeezed down to a single Q-value at the end
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        request_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        request_features: [num_requests, request_feature_dim]
        vehicle_features:  [num_vehicles, vehicle_feature_dim]
        edge_index:        [2, num_edges], node order must match MatchGraph
                            (request nodes first, then vehicle nodes)

        Returns a single scalar tensor: Q(s,a) for this whole graph. Output
        is left unbounded on purpose (no sigmoid/clamp) even though the
        Monte-Carlo return target is a service rate in [0,1] - a plain
        linear head plus Huber loss should learn to land in that range
        without the vanishing gradients sigmoid saturation would cause.
        """
        if request_features.ndim != 2 or vehicle_features.ndim != 2:
            raise ValueError("request_features and vehicle_features must be 2D.")

        num_requests = request_features.shape[0]

        # encode both node types into the same hidden_dim space and stack
        # them - requests first, then vehicles, matching MatchGraph's node order
        request_embeddings = self.request_encoder(request_features)
        vehicle_embeddings = self.vehicle_encoder(vehicle_features)
        node_embeddings = torch.cat([request_embeddings, vehicle_embeddings], dim=0)

        # message passing: each node repeatedly mixes in its neighbours' embeddings
        edge_index = edge_index.to(node_embeddings.device)
        for layer in self.message_passing_layers:
            node_embeddings = layer(node_embeddings, edge_index)

        # pooling: mean over requests and mean over vehicles, kept separate so
        # neither node count nor node type gets blended away before the head
        request_pooled = node_embeddings[:num_requests].mean(dim=0)
        vehicle_pooled = node_embeddings[num_requests:].mean(dim=0)
        graph_embedding = torch.cat([request_pooled, vehicle_pooled], dim=0)

        # head: pooled graph vector -> one scalar Q-value
        q_value = self.q_head(graph_embedding).squeeze(-1)
        return q_value
