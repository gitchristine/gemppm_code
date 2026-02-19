"""
DATL — Domain-Aware Triplet Loss
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DATLEncoder(nn.Module):

    def __init__(
        self,
        num_activities: int,
        feature_dim: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.3,
    ):

        super().__init__()

        self.d_model = d_model

        # Embed activities
        self.activity_emb = nn.Embedding(num_activities, d_model, padding_idx=0)

        # Project numerical features to d_model
        self.feature_proj = nn.Linear(feature_dim, d_model)

        # Combine activity embedding + feature projection
        self.combine = nn.Linear(2 * d_model, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Layer norm on output
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        activities: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:

        # Activity embeddings
        act_emb = self.activity_emb(activities)  # (B, S, D)

        # Feature projection
        feat_emb = self.feature_proj(features)   # (B, S, D)

        # Combine
        combined = torch.cat([act_emb, feat_emb], dim=-1)
        x = self.combine(combined)  # (B, S, D)

        # Positional encoding
        x = self.pos_enc(x)

        # Padding mask (True = ignore)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, S, D)

        # Pool: mean of non-padded positions
        mask_float = (~mask).float().unsqueeze(-1)  # (B, S, 1)
        h = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)  # (B, D)
        h = self.out_norm(h)

        return h

    def embedding_dim(self) -> int:
        return self.d_model


class TripletLoss(nn.Module):

    def __init__(self, margin: float = 0.3, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance == "cosine":
            d_ap = 1.0 - F.cosine_similarity(anchor, positive)
            d_an = 1.0 - F.cosine_similarity(anchor, negative)
        else:
            d_ap = (anchor - positive).pow(2).sum(dim=1).sqrt()
            d_an = (anchor - negative).pow(2).sum(dim=1).sqrt()

        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()
