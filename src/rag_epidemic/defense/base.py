"""Defense interfaces. All defenses get the same signature so they can be
swapped in the orchestrator."""

from __future__ import annotations

from typing import Protocol

from ..rag.retrievers import RetrievedChunk


class Defense(Protocol):
    name: str

    def filter(
        self,
        query: str,
        query_embedding,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        ...

    def update(self, **kwargs) -> None:
        ...
