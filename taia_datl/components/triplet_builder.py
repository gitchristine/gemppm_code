"""
Triplet Builder  [*]
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class TripletBuilder:
    """
    Build triplet batches from a FAISS (or random) index.
    TODONE solve edge case: no valid triplets
    """

    def __init__(self, index, top_k: int = 5, seed: int = 42):

        self.index = index
        self.top_k = top_k
        self.rng = np.random.RandomState(seed)

    def build_triplets(
        self,
        embeddings: np.ndarray,
        domain_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        distances, indices, _, nbr_domains = self.index.search(
            embeddings, self.top_k
        )

        anchors, positives, negatives = [], [], []

        for i in range(len(embeddings)):
            my_domain = domain_ids[i]
            pos_idx = None
            neg_idx = None

            for j in range(self.top_k):
                nbr_i = indices[i, j]
                nbr_d = nbr_domains[i][j]

                if nbr_i == i:
                    continue

                if nbr_d == my_domain and pos_idx is None:
                    pos_idx = nbr_i
                elif nbr_d != my_domain and neg_idx is None:
                    neg_idx = nbr_i

                if pos_idx is not None and neg_idx is not None:
                    break

            # Fallback: random sampling if neighbours were insufficient
            if pos_idx is None:
                same_mask = (domain_ids == my_domain)
                same_mask[i] = False
                candidates = np.where(same_mask)[0]
                if len(candidates) > 0:
                    pos_idx = self.rng.choice(candidates)
                else:
                    continue  # skip this

            if neg_idx is None:
                diff_mask = (domain_ids != my_domain)
                candidates = np.where(diff_mask)[0]
                if len(candidates) > 0:
                    neg_idx = self.rng.choice(candidates)
                else:
                    continue  # skip this

            anchors.append(i)
            positives.append(pos_idx)
            negatives.append(neg_idx)

        if not anchors:

            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

        return (
            np.array(anchors, dtype=int),
            np.array(positives, dtype=int),
            np.array(negatives, dtype=int),
        )
