"""
Simple scalar ensemble for remaining-time fusion.

rt_final = β × rt_direct + (1 - β) × rt_retrieved

β is a fixed scalar set in TAIADATLConfig.fusion_beta and tuned on the
validation set.

# Future work -> the parameters could be learnt based on the dataset that is being used.
"""

from __future__ import annotations

import torch


class FusionGate:
    """Stateless scalar blend of direct and retrieved remaining-time estimates.

    Args:
        beta: Weight given to the direct (TinyLLM head) prediction.
              Must be in [0, 1].  The retrieved (FAISS) estimate receives
              weight (1 - beta).
    """

    def __init__(self, beta: float = 0.5):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta

    def fuse(
        self,
        rt_direct: torch.Tensor,
        rt_retrieved: torch.Tensor,
    ) -> torch.Tensor:
        """Blend the two remaining-time estimates.

        Args:
            rt_direct:    (batch,) predictions from TimeHead (TinyLLM backbone).
            rt_retrieved: (batch,) estimates from FAISS nearest-neighbour lookup.

        Returns:
            rt_final: (batch,) weighted combination.
        """
        return self.beta * rt_direct + (1.0 - self.beta) * rt_retrieved