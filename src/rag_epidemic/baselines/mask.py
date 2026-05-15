"""M-ASK: Misuse-Aware Source Knowledge; per-author trust score."""
from __future__ import annotations

from collections import defaultdict

from ..rag.retrievers import RetrievedChunk


class MASK:
    name = "mask"

    def __init__(self, init_trust: float = 1.0, decay: float = 0.05, threshold: float = 0.4):
        self.init = init_trust
        self.decay = decay
        self.threshold = threshold
        self.trust: dict[str, float] = defaultdict(lambda: init_trust)

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        return [c for c in candidates
                if self.trust[c.metadata.get("author_agent", "?")] >= self.threshold]

    def update(self, **kwargs):
        ev = kwargs.get("event")
        if not ev:
            return
        if ev.get("type") == "verifier_flag":
            a = ev["agent_id"]
            self.trust[a] = max(0.0, self.trust[a] - self.decay)
        elif ev.get("type") == "verifier_confirm":
            a = ev["agent_id"]
            self.trust[a] = min(self.init, self.trust[a] + self.decay * 0.5)
