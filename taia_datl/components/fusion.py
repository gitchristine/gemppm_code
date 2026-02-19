"""
FAISS-Conditioned Adaptive Fusion Gate  [*]

TODO loss function placement?
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionGate(nn.Module):
    def __init__(
        self,
        datl_dim: int = 256,
        taia_dim: int = 256,
        hidden_dim: int = 64,
    ):

        super().__init__()

        # Input: dist_k (scalar) + h_datl + h_taia
        input_dim = 1 + datl_dim + taia_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # α ∈ [0, 1]
        )

    def forward(
        self,
        dist_k: torch.Tensor,
        h_datl: torch.Tensor,
        h_taia: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat([dist_k, h_datl, h_taia], dim=-1)
        alpha = self.mlp(x)
        return alpha

    @staticmethod
    def fuse(
        alpha: torch.Tensor,
        p_taia: torch.Tensor,
        p_datl: torch.Tensor,
    ) -> torch.Tensor:

        return alpha * p_taia + (1.0 - alpha) * p_datl
