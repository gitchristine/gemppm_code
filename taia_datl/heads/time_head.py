"""
Remaining-Time Prediction Head
================================
Maps a trace embedding h ∈ ℝ^D to a scalar remaining-time estimate.

    t̂ = ReLU(W₂ · ReLU(W₁ · h))

The final ReLU ensures non-negative predictions (time cannot be negative).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeHead(nn.Module):
    """MLP head: embedding → remaining-time scalar."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # remaining time ≥ 0
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) trace embedding.
        Returns:
            time_pred: (batch, 1) predicted remaining time.
        """
        return self.head(h)
