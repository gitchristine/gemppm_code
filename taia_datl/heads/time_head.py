"""
Remaining-Time Prediction Head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeHead(nn.Module):
    """Gated FFN head: embedding -> remaining-time scalar."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # dropout retained in signature for API compatibility
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)  # gate branch
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)  # value branch
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False) # output projection
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) trace embedding from TinyLLM.
        Returns:
            time_pred: (batch, 1) predicted remaining time.
        """
        gated = F.silu(self.W1(h)) * self.W2(h)        # SiLU gate ⊙ value
        h_ffn = self.norm(self.W3(gated))              # project + normalise
        return F.relu(self.out(h_ffn))                 # non-negative output