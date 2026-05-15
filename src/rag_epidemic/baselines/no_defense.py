"""No-op defense (baseline 'undefended')."""
from __future__ import annotations

from ..rag.retrievers import RetrievedChunk


class NoDefense:
    name = "undefended"

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        return list(candidates)

    def update(self, **kwargs):
        return None
