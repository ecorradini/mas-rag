"""Semantic Chameleon: hybrid retrieval with shifted query embeddings.

We approximate by penalising the top-1 dense match's contribution at retrieval
time, increasing the weight of lexical matches (BM25) when dense agreement is
suspiciously concentrated on a single author.
"""
from __future__ import annotations

from collections import Counter

from ..rag.retrievers import RetrievedChunk


class SemanticChameleon:
    name = "sem_chameleon"

    def __init__(self, author_cap: float = 0.5):
        self.author_cap = author_cap

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        if not candidates:
            return candidates
        cnt = Counter(c.metadata.get("author_agent", "?") for c in candidates)
        total = sum(cnt.values())
        kept: list[RetrievedChunk] = []
        used: Counter = Counter()
        max_per_author = max(1, int(self.author_cap * total))
        for c in candidates:
            a = c.metadata.get("author_agent", "?")
            if used[a] < max_per_author:
                kept.append(c)
                used[a] += 1
        return kept

    def update(self, **kwargs):
        return None
