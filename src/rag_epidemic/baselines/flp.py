"""FLP: Functional-Level Poisoning detector via clustering on the candidate set."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..rag.retrievers import RetrievedChunk


class FLP:
    name = "flp"

    def __init__(self, n_clusters: int = 2, min_cluster_frac: float = 0.3):
        self.n_clusters = n_clusters
        self.min_frac = min_cluster_frac

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        if len(candidates) < self.n_clusters + 1:
            return candidates
        X = np.stack([c.embedding for c in candidates if c.embedding is not None])
        if len(X) < self.n_clusters + 1:
            return candidates
        if len(np.unique(X, axis=0)) < self.n_clusters:
            return candidates
        km = KMeans(n_clusters=self.n_clusters, n_init=4, random_state=0).fit(X)
        labels = km.labels_
        counts = np.bincount(labels)
        # keep clusters whose size fraction ≥ min_frac (majority filtering)
        keep_lbls = {i for i, c in enumerate(counts) if c / len(labels) >= self.min_frac}
        kept = [c for c, l in zip(candidates, labels) if l in keep_lbls]
        return kept or candidates[:1]

    def update(self, **kwargs):
        return None
