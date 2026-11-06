"""PyTorch MLP classifier."""

import torch
from torch import nn

from src import config


class MovementClassifier(nn.Module):
    """MLP: [Linear -> BatchNorm -> ReLU -> Dropout] per hidden layer -> Linear.

    input_dim: 76 (IMU-only), 396 (pose-only) or 472 (fused).
    hidden_dims=[] gives a single Linear layer (multinomial logistic regression).
    """

    def __init__(self, input_dim: int, num_classes: int = len(config.SELECTED_ACTIONS),
                 hidden_dims: list[int] = config.HIDDEN_DIMS, dropout: float = config.DROPOUT):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of feature vectors."""
        return self.net(x)  # logits


# Named architecture variants: label -> MovementClassifier kwargs.
MODEL_VARIANTS = {
    "MLP (256-128, dp=0.3)": {"hidden_dims": [256, 128], "dropout": 0.3},
    "MLP (128, dp=0.5)": {"hidden_dims": [128], "dropout": 0.5},
    "MLP (64, dp=0.5)": {"hidden_dims": [64], "dropout": 0.5},
    "Linear (logistic reg.)": {"hidden_dims": [], "dropout": 0.0},
}
