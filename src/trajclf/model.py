"""1D CNN classifying a window of IMU signal."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class RideCNN(nn.Module):
    """Input (batch, window, channels) -> logits (batch, n_classes)."""

    def __init__(self, n_channels: int = 6, n_classes: int = 5, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(n_channels, width, 7),
            _block(width, width * 2, 5),
            _block(width * 2, width * 4, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4 * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.transpose(1, 2))  # Conv1d expects (batch, channels, time)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # mean + max over time
        return self.head(pooled)
