"""
FAISS Persistent Triplet Index  [*]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FAISSIndex:

    def __init__(
        self,
        dim: int = 256,
        index_type: str = "flat",
        nprobe: int = 10,
        save_dir: str = "faiss_indices",
    ):

        if not HAS_FAISS:
            raise ImportError("faiss-cpu or faiss-gpu is required: "
                              "pip install faiss-cpu")

        self.dim = dim
        self.index_type = index_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Build the index
        if index_type == "ivf":
            quantiser = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantiser, dim, min(64, 4))
            self.index.nprobe = nprobe
        else:
            self.index = faiss.IndexFlatL2(dim)

        # Metadata parallel array
        self._ids: list[str] = []  # case_id for each vector
        self._labels: list[int] = []  # domain_id for each vector

        print(f"[FAISS] Initialised {index_type} index  dim={dim}")

    # ------------------------------------------------------------------
    # Build / add
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        case_ids: list[str],
        domain_ids: list[int],
    ) -> None:


        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        assert embeddings.shape[1] == self.dim
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        self.index.add(embeddings)
        self._ids.extend(case_ids)
        self._labels.extend(domain_ids)

        print(f"[FAISS] Added {len(case_ids)} vectors  (total={self.index.ntotal})")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        distances, indices = self.index.search(query, top_k)

        # Map indices to metadata
        out_case_ids = []
        out_domain_ids = []
        for row in indices:
            out_case_ids.append([self._ids[i] if 0 <= i < len(self._ids) else "" for i in row])
            out_domain_ids.append([self._labels[i] if 0 <= i < len(self._labels) else -1 for i in row])

        return distances, indices, out_case_ids, out_domain_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "triplet_index") -> None:
        """Write the index + metadata to disk."""
        faiss.write_index(self.index, str(self.save_dir / f"{name}.faiss"))
        np.savez(
            self.save_dir / f"{name}_meta.npz",
            ids=np.array(self._ids, dtype=object),
            labels=np.array(self._labels, dtype=np.int32),
        )
        print(f"[FAISS] Saved index → {self.save_dir / name}.faiss")

    def load(self, name: str = "triplet_index") -> None:
        """Load the index + metadata from disk."""
        idx_path = self.save_dir / f"{name}.faiss"
        meta_path = self.save_dir / f"{name}_meta.npz"

        self.index = faiss.read_index(str(idx_path))
        meta = np.load(str(meta_path), allow_pickle=True)
        self._ids = meta["ids"].tolist()
        self._labels = meta["labels"].tolist()
        print(f"[FAISS] Loaded index ({self.index.ntotal} vectors) from {idx_path}")


class RandomFallbackIndex:

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._embeddings: Optional[np.ndarray] = None
        self._ids: list[str] = []
        self._labels: list[int] = []

    def build(self, embeddings: np.ndarray, case_ids: list, domain_ids: list):
        self._embeddings = embeddings
        self._ids = list(case_ids)
        self._labels = list(domain_ids)
        print(f"[RandomIndex] Stored {len(case_ids)} vectors (FAISS disabled)")

    def search(self, query: np.ndarray, top_k: int = 5):
        n = len(self._ids)
        q = query.shape[0]
        indices = np.array([self.rng.choice(n, size=top_k, replace=False) for _ in range(q)])
        distances = np.ones_like(indices, dtype=np.float32)
        case_ids = [[self._ids[i] for i in row] for row in indices]
        domain_ids = [[self._labels[i] for i in row] for row in indices]
        return distances, indices, case_ids, domain_ids

    def save(self, name="triplet_index"):
        pass  # no-op

    def load(self, name="triplet_index"):
        pass  # no-op
