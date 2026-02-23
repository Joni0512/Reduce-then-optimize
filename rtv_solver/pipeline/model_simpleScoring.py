import torch
import torch.nn as nn


class ScoringMLP(nn.Module):
    """
    Item-wise scoring network for structured prediction.

    A shared MLP is applied independently to each item's feature vector,
    producing one raw score per item. The number of items is variable;
    the feature dimension is fixed for the lifetime of the model.

    Input:
        x : Tensor of shape (num_items, feature_dim)
            or (batch_size, num_items, feature_dim) for mini-batch use.

    Output:
        Tensor of shape (num_items,) or (batch_size, num_items).
        Raw scores — no activation applied to the final layer.
    """

    def __init__(self, feature_dim: int = 0, hidden_dim: int = 64) -> None:
        """
        Args:
            feature_dim: Number of features per item. Must be constant
                         across all instances seen during a training run.
            hidden_dim:  Width of the single hidden layer (default 64).
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # added more complexity to the model to test if it can easily reach convergence
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., num_items, feature_dim)

        Returns:
            scores: (..., num_items)  — raw, unbounded scores.
        """
        return self.net(x).squeeze(-1)
