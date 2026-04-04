"""
Triplet Builder  [*]
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class TripletBuilder:
    """
    Build triplet batches from a FAISS (or random) index.
    """

    def __init__(self, index, top_k: int = 5, seed: int = 42):

        self.index = index
        self.top_k = top_k
        self.rng = np.random.RandomState(seed)

    def build_triplets(
        self,
        embeddings: np.ndarray,
        domain_ids: np.ndarray,
        rt_buckets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build domain-aware triplets (D-Triplet formulation).

        Positive: same RT bucket, different domain.
        Negative: different RT bucket, same domain.
        """
        distances, indices, _, nbr_domains, nbr_rt_buckets, _ = self.index.search(
            embeddings, self.top_k
        )

        anchors, positives, negatives = [], [], []

        for i in range(len(embeddings)):
            my_domain = domain_ids[i]
            my_rt_bucket = rt_buckets[i]
            pos_idx = None
            neg_idx = None

            for j in range(self.top_k):
                nbr_i = indices[i, j]
                nbr_d = nbr_domains[i][j]
                nbr_b = nbr_rt_buckets[i][j]

                if nbr_i == i:
                    continue

                # Positive: same RT bucket, different domain
                if nbr_b == my_rt_bucket and nbr_d != my_domain and pos_idx is None:
                    pos_idx = nbr_i
                # Negative: different RT bucket, same domain
                elif nbr_b != my_rt_bucket and nbr_d == my_domain and neg_idx is None:
                    neg_idx = nbr_i

                if pos_idx is not None and neg_idx is not None:
                    break

            # Fallback: random sampling if neighbours were insufficient
            if pos_idx is None:
                # same RT bucket, different domain
                same_b = (rt_buckets == my_rt_bucket)
                diff_d = (domain_ids != my_domain)
                candidates = np.where(same_b & diff_d)[0]
                if len(candidates) > 0:
                    pos_idx = self.rng.choice(candidates)
                else:
                    continue  # skip this anchor

            if neg_idx is None:
                # different RT bucket, same domain
                diff_b = (rt_buckets != my_rt_bucket)
                same_d = (domain_ids == my_domain)
                candidates = np.where(diff_b & same_d)[0]
                if len(candidates) > 0:
                    neg_idx = self.rng.choice(candidates)
                else:
                    continue  # skip this anchor

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