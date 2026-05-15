"""SeCon-RAG: source-consensus filter.

A chunk passes only if there exist ≥k peer chunks (different author)
with cosine similarity ≥ τ_sim that semantically agree.
"""
from __future__ import annotations

import numpy as np

from ..rag.retrievers import RetrievedChunk


class SeConRAG:
    name = "secon_rag"

    def __init__(self, k: int = 2, tau_sim: float = 0.6):
        self.k = k
        self.tau = tau_sim

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        kept: list[RetrievedChunk] = []
        embs = [c.embedding for c in candidates]
        for i, c in enumerate(candidates):
            if c.embedding is None:
                kept.append(c)
                continue
            peers = 0
            for j, d in enumerate(candidates):
                if i == j or d.embedding is None:
                    continue
                if d.metadata.get("author_agent") == c.metadata.get("author_agent"):
                    continue
                sim = float(np.dot(c.embedding, d.embedding) /
                            (np.linalg.norm(c.embedding) * np.linalg.norm(d.embedding) + 1e-8))
                if sim >= self.tau:
                    peers += 1
            if peers >= self.k:
                kept.append(c)
        return kept or candidates[:1]  # keep at least 1 to avoid empty context

    def update(self, **kwargs):
        return None
