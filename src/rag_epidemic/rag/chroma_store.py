"""ChromaDB-backed vector store with rich metadata.

We use the persistent client per-run, in a namespaced collection. Each chunk
carries: chunk_id, author_agent, sim_event_id, channel, t (superstep),
is_poisoned (ground truth), text, embedding.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore


@dataclass
class Chunk:
    chunk_id: str
    text: str
    author_agent: str
    sim_event_id: str | None
    channel: str
    t: int
    is_poisoned: bool
    embedding: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ChromaStore:
    """Wrapper around a single ChromaDB collection.

    We do *not* let Chroma compute embeddings; we always supply our own
    so the embedding model is interchangeable.
    """

    def __init__(self, path: str | Path, name: str = "rage", reset: bool = False):
        if chromadb is None:
            raise ImportError("chromadb is required")
        self.path = Path(path)
        if reset and self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        try:
            self.col = self.client.get_collection(name)
        except Exception:
            self.col = self.client.create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )

    # ---------------- write ----------------
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        embs = [c.embedding.tolist() if c.embedding is not None else None for c in chunks]
        metas = [
            {
                "author_agent": c.author_agent,
                "sim_event_id": c.sim_event_id or "",
                "channel": c.channel,
                "t": int(c.t),
                "is_poisoned": bool(c.is_poisoned),
                **{f"x_{k}": v for k, v in c.extra.items()},
            }
            for c in chunks
        ]
        if any(e is None for e in embs):
            raise ValueError("All chunks must have precomputed embeddings.")
        self.col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    # ---------------- read ----------------
    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        res = self.col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        out: list[dict[str, Any]] = []
        for i in range(len(res["ids"][0])):
            out.append(
                {
                    "chunk_id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "distance": float(res["distances"][0][i]),
                    "similarity": 1.0 - float(res["distances"][0][i]),
                    "embedding": np.asarray(res["embeddings"][0][i], dtype=np.float32),
                }
            )
        return out

    def count(self) -> int:
        return int(self.col.count())

    def get_all(self) -> dict[str, Any]:
        return self.col.get(include=["documents", "metadatas", "embeddings"])

    def update_metadata(self, chunk_id: str, **fields: Any) -> None:
        cur = self.col.get(ids=[chunk_id], include=["metadatas"])
        if not cur["ids"]:
            return
        meta = dict(cur["metadatas"][0])
        meta.update(fields)
        self.col.update(ids=[chunk_id], metadatas=[meta])
