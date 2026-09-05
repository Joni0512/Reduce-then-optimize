from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rtv_solver.pipeline.candidate_scoring_gnn import (
    GCNMeanLayer,
    GraphSAGEMeanLayer,
    GraphSAGEPoolLayer,
)
from rtv_solver.pipeline.match_graph_features import MatchGraphFeatureBuilder


class GATLayer(nn.Module):
    """
    2026-09-05: Graph Attention layer (Velickovic et al., ICLR 2018) - see
    chat and docs/SRL_Design.md's GAT plan section for the full derivation.
    Learns a per-edge attention weight instead of a fixed mean/max like the
    other three layers, so a request/vehicle can weigh its neighbours by
    learned importance rather than treating every neighbour equally.

    Unlike GCNMeanLayer/GraphSAGEMeanLayer/GraphSAGEPoolLayer, this layer's
    output dimension is NOT always hidden_dim: `is_final_layer=False` (used
    for every layer except the last) concatenates the num_heads independent
    attention heads (Eq. 5 in the paper), giving an output of
    num_heads*out_dim; `is_final_layer=True` averages them instead (Eq. 6),
    keeping the output at out_dim. CriticGNN's construction code accounts
    for this when chaining GATLayer instances, unlike the uniform
    `layer_cls(hidden_dim, dropout=dropout)` loop used for the other three
    aggregators.

    No residual connection / LayerNorm (unlike the other three layers) -
    matches the original GAT paper, which does not use one in this basic
    layer. A residual connection is not directly possible here anyway
    without a projection, since in_dim != out_dim whenever this layer
    concatenates (every non-final layer) or follows a concatenating layer
    (the final layer's in_dim is num_heads*out_dim from the previous layer).
    Discussed 2026-09-05 (see chat): a residual WITH a learned linear
    projection (in_dim -> out_dim of this layer) to make the shapes match
    was considered as an alternative, but decided against for this first
    version to stay paper-consistent and keep the first gat-vs-gcn ablation
    as simple as possible.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_heads: int = 4,
        is_final_layer: bool = False,
        leaky_relu_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_heads < 1:
            raise ValueError("num_heads must be at least one.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.is_final_layer = is_final_layer

        # one shared linear transform W^k and one attention vector a^k per
        # head - kept as separate nn.Linear instances (not one batched
        # tensor) so each head's parameters are independently initialised
        # and trained, matching the paper's "K independent attention
        # mechanisms" (Eq. 5).
        self.head_transforms = nn.ModuleList(
            nn.Linear(in_dim, out_dim) for _ in range(num_heads)
        )
        # a^T [W h_i || W h_j] implemented as one Linear(2*out_dim -> 1) per
        # head, equivalent to a single-layer feedforward network with weight
        # vector a in R^{2*out_dim} (paper, Section 2.1).
        self.attn_scores = nn.ModuleList(
            nn.Linear(2 * out_dim, 1, bias=False) for _ in range(num_heads)
        )
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _scatter_softmax(scores: torch.Tensor, target_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        softmax_j(e_ij) normalised per target node i (Eq. 2), computed over
        an edge list instead of a dense adjacency matrix. Subtracts the
        per-target max first for numerical stability (standard softmax
        trick), same idea as GraphSAGEPoolLayer's index_reduce_(..., "amax")
        usage above.
        """
        max_per_target = scores.new_full((num_nodes,), float("-inf"))
        max_per_target = max_per_target.scatter_reduce(
            0, target_index, scores, reduce="amax", include_self=True
        )
        shifted = scores - max_per_target[target_index]
        exp_scores = shifted.exp()
        sum_per_target = scores.new_zeros(num_nodes)
        sum_per_target.index_add_(0, target_index, exp_scores)
        return exp_scores / sum_per_target[target_index].clamp_min(1e-16)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [num_nodes, hidden_dim].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        num_nodes = node_embeddings.shape[0]
        device = node_embeddings.device

        # N_i = edge_index neighbours + self (see chat/docs/SRL_Design.md) -
        # self-loops added here so every node always attends to itself too,
        # including isolated nodes with zero edges (guarantees N_i is never
        # empty, so the softmax below never divides by zero).
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        aug_edge_index = torch.cat([edge_index.to(device), self_loops], dim=1)
        source_nodes = aug_edge_index[0]  # j
        target_nodes = aug_edge_index[1]  # i

        head_outputs = []
        for head_idx in range(self.num_heads):
            transformed = self.head_transforms[head_idx](node_embeddings)  # W^k h, [N, out_dim]

            transformed_i = transformed[target_nodes]  # W^k h_i per edge
            transformed_j = transformed[source_nodes]  # W^k h_j per edge
            concat_ij = torch.cat([transformed_i, transformed_j], dim=-1)  # [E, 2*out_dim]

            raw_scores = self.attn_scores[head_idx](concat_ij).squeeze(-1)  # e_ij, [E]
            raw_scores = self.leaky_relu(raw_scores)

            attention_weights = self._scatter_softmax(raw_scores, target_nodes, num_nodes)  # alpha_ij

            weighted_messages = attention_weights.unsqueeze(-1) * transformed_j  # alpha_ij * W^k h_j
            head_output = transformed.new_zeros((num_nodes, self.out_dim))
            head_output.index_add_(0, target_nodes, weighted_messages)  # sum_j alpha_ij * W^k h_j

            head_outputs.append(head_output)

        if self.is_final_layer:
            # Eq. 6: average raw per-head outputs FIRST, apply sigma ONCE
            # afterwards (concatenation would no longer make sense here,
            # since this layer's output must stay at out_dim for whatever
            # comes next, e.g. CriticGNN's pooling + q_head).
            averaged = torch.stack(head_outputs, dim=0).mean(dim=0)
            updated = F.relu(averaged)
        else:
            # Eq. 5: apply sigma per head, THEN concatenate - output is
            # num_heads*out_dim wide, fed into the next GATLayer.
            updated = torch.cat([F.relu(h) for h in head_outputs], dim=-1)

        return self.dropout(updated)


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
    2026-09-05: attention (aggregator="gat") added - see GATLayer above and
    docs/SRL_Design.md's GAT plan section; target network / twin critic still
    not implemented here.
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
        gat_num_heads: int = 4,
    ) -> None:
        super().__init__()

        if aggregator not in ("gcn", "mean", "pool", "gat"):
            raise ValueError(f"aggregator must be 'gcn', 'mean', 'pool', or 'gat', got {aggregator!r}.")
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
        if aggregator == "gat":
            # 2026-09-05: GATLayer's output dimension is NOT always
            # hidden_dim (see its docstring) - every layer except the last
            # concatenates gat_num_heads heads (output = gat_num_heads *
            # hidden_dim), the last layer averages them back down to
            # hidden_dim so q_head's expected input size is unaffected.
            # Unlike the other three aggregators, in_dim therefore varies
            # per layer, so this can't reuse the uniform
            # `layer_cls(hidden_dim, dropout=dropout)` loop below.
            gat_layers = []
            current_dim = hidden_dim
            for layer_idx in range(num_message_passing_layers):
                is_final = layer_idx == num_message_passing_layers - 1
                gat_layers.append(
                    GATLayer(
                        current_dim,
                        hidden_dim,
                        num_heads=gat_num_heads,
                        is_final_layer=is_final,
                        dropout=dropout,
                    )
                )
                current_dim = hidden_dim if is_final else gat_num_heads * hidden_dim
            self.message_passing_layers = nn.ModuleList(gat_layers)
        else:
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
