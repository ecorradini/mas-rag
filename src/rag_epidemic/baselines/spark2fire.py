"""Spark2Fire: rate-limit individual authors when their write frequency
exceeds a moving threshold (firewall-style)."""
from __future__ import annotations

from collections import defaultdict, deque

from ..rag.retrievers import RetrievedChunk


class Spark2Fire:
    name = "spark2fire"

    def __init__(self, window: int = 20, max_writes: int = 5):
        self.window = window
        self.max_writes = max_writes
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.blocked: set[str] = set()

    def observe_write(self, author: str, t: int) -> None:
        self.history[author].append(t)
        if len(self.history[author]) >= self.max_writes:
            span = t - self.history[author][0]
            if span <= self.window:
                self.blocked.add(author)

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        return [c for c in candidates
                if c.metadata.get("author_agent") not in self.blocked]

    def update(self, **kwargs):
        ev = kwargs.get("event")
        if ev and ev.get("type") == "chunk_written":
            self.observe_write(ev["agent_id"], int(ev.get("t", 0)))
