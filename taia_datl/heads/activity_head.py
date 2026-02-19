"""
Activity Prediction Head
=========================
Maps a trace embedding h ∈ ℝ^D to a distribution over next activities.

    p(next | h) = softmax(W₂ · ReLU(W₁ · h))

Used on top of the DATL encoder (and optionally fused with TAIA output).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActivityHead(nn.Module):
    """MLP head: embedding → next-activity logits."""

    def __init__(self, input_dim: int, num_activities: int, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_activities),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) trace embedding.
        Returns:
            logits: (batch, num_activities).
        """
        return self.head(h)
