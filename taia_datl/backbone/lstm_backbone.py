"""
LSTMBackbone
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBackbone(nn.Module):

    def __init__(
        self,
        num_activities: int,
        feature_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self._output_dim = hidden_dim * 2  # bidirectional

        # ----------------------------------------------------------------
        # Input projections
        # Each activity/feature is projected to hidden_dim so they can be
        # summed before the LSTM (avoids doubling the LSTM input width).
        # ----------------------------------------------------------------
        self.activity_emb = nn.Embedding(num_activities, hidden_dim, padding_idx=0)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        # Fuse the two hidden_dim vectors into one
        self.input_combine = nn.Linear(2 * hidden_dim, hidden_dim)

        # ----------------------------------------------------------------
        # Bidirectional LSTM
        # ----------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ----------------------------------------------------------------
        # Additive (Bahdanau-style) attention pooling
        # Lets the model focus on the most informative time steps instead
        # of naively averaging all positions.
        # ----------------------------------------------------------------
        self.attn_U = nn.Linear(self._output_dim, hidden_dim, bias=False)
        self.attn_W = nn.Linear(hidden_dim, 1, bias=False)

        # ----------------------------------------------------------------
        # Output regularisation
        # ----------------------------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(self._output_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        activities: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = activities.shape
        act_emb  = self.activity_emb(activities)            # (B, S, H)
        feat_emb = self.feature_proj(features)              # (B, S, H)
        x = self.input_combine(
            torch.cat([act_emb, feat_emb], dim=-1)
        )                                                   # (B, S, H)
        x = F.relu(x)
        x = self.dropout(x)

        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )

        energy = self.attn_W(
            torch.tanh(self.attn_U(lstm_out))
        ).squeeze(-1)                                       # (B, S)

        pad_mask = (
            torch.arange(seq_len, device=activities.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        energy = energy.masked_fill(pad_mask, float("-inf"))
        alpha = torch.softmax(energy, dim=1).unsqueeze(-1)  # (B, S, 1)


        h = (alpha * lstm_out).sum(dim=1)                   # (B, 2·H)

        # Normalise and return
        h = self.dropout(h)
        h = self.out_norm(h)

        return h


    def embedding_dim(self) -> int:
        return self._output_dim