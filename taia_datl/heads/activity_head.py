"""
Activity Prediction Head
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActivityHead(nn.Module):
    """Linear classifier: embedding -> next-activity logits."""

    def __init__(self, input_dim: int, num_activities: int, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        # hidden_dim and dropout retained in signature for API compatibility
        self.head = nn.Linear(input_dim, num_activities)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) trace embedding from TinyLLM.
        Returns:
            logits: (batch, num_activities).
        """
        return self.head(h)