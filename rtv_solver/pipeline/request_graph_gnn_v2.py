import torch
import torch.nn as nn


class RequestGraphEdgeGNNv2(nn.Module):
    """
    Deeper variant of RequestGraphEdgeGNN.

    Differences vs. the original:
        - one distinct message/update MLP per message-passing layer
          (no weight sharing across layers)
        - residual connection around each node update
        - LayerNorm after each node/edge update
        - edge embeddings are updated every layer, not just once
        - optional dropout for regularization

    Input/output signature is identical to RequestGraphEdgeGNN so it is a
    drop-in replacement in RequestGraphPruner and the training scripts.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # One independent set of weights per layer (no weight tying).
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.node_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Edge embeddings are refreshed each layer from their own endpoints.
        self.edge_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.dim() != 2:
            raise ValueError("edge_index must be a 2D tensor.")

        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = x.shape[0]

        for layer in range(self.num_layers):
            h_src = h[src]
            h_dst = h[dst]

            message_input = torch.cat([h_src, h_dst, e], dim=-1)
            messages = self.message_mlps[layer](message_input)

            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, dst, messages)

            reverse_aggregated = torch.zeros_like(h)
            reverse_aggregated.index_add_(0, src, messages)

            aggregated = aggregated + reverse_aggregated

            degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            degree.index_add_(0, src, torch.ones_like(src, dtype=x.dtype))
            degree = degree.clamp(min=1.0).unsqueeze(-1)

            aggregated = aggregated / degree

            # Residual node update + LayerNorm.
            update_input = torch.cat([h, aggregated], dim=-1)
            h = self.node_norms[layer](h + self.node_updates[layer](update_input))

            # Residual edge update + LayerNorm, using the freshly updated node embeddings.
            h_src = h[src]
            h_dst = h[dst]
            edge_update_input = torch.cat([h_src, h_dst, e], dim=-1)
            e = self.edge_norms[layer](e + self.edge_updates[layer](edge_update_input))

        h_src = h[src]
        h_dst = h[dst]

        edge_score_input = torch.cat([h_src, h_dst, e], dim=-1)

        # Raw logits: BCEWithLogitsLoss applies sigmoid internally.
        logits = self.edge_scorer(edge_score_input).squeeze(-1)

        return logits
