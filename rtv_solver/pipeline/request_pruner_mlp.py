"""
request_pruner_mlp.py
======================
2026-07-14: Phase 1 (model) of the request-pruner design discussion.

Plain feed-forward node classifier for request-only pruning - deliberately
NOT a GNN. request_pruner_signal_check.py already showed that a single
LINEAR layer on the 15 raw features reaches ROC-AUC ~0.83 on held-out
instances (see that script's results from 2026-07-13/14, validated on both
the provisional 12-instance split and the real 56-instance mixed_all
split). The open question this model answers is whether non-linear
interactions BETWEEN those 15 features add anything on top of that - not
whether a request needs information from its neighbors (that's a separate,
harder question the request-request pair pruner already handles via message
passing; deliberately kept out of scope here, see the design discussion for
why: request pruning and pair pruning are meant to be independently testable
first).

Because the linear baseline is already strong and the dataset is fairly
small (~6.9k train rows across 36 instances, many rows correlated within
the same rolling-horizon window), this model is kept intentionally small -
1-3 hidden layers, hidden_dim in the 16-64 range. Going deeper/wider risks
overfitting without a clear reason to expect more signal is there to find.
"""

import torch
import torch.nn as nn


class RequestPrunerMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        num_hidden_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw logits, one per row (request). BCEWithLogitsLoss applies
        # sigmoid internally, same convention as RequestGraphEdgeGNN(v2).
        return self.network(x).squeeze(-1)
