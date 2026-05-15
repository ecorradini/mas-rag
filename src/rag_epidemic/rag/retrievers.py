"""Retrievers: dense (Chroma) and a hybrid BM25+dense for Semantic Chameleon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .chroma_store import ChromaStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float
    embedding: np.ndarray | None = None


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in text.replace("\n", " ").split() if t]


class DenseRetriever:
    def __init__(self, store: ChromaStore):
        self.store = store

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> list[RetrievedChunk]:
        rows = self.store.query(query_embedding, k=k)
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                metadata=r["metadata"],
                similarity=r["similarity"],
                embedding=r.get("embedding"),
            )
            for r in rows
        ]


class HybridBM25DenseRetriever:
    """λ-weighted hybrid retriever used by Semantic Chameleon baseline."""

    def __init__(self, store: ChromaStore, lam: float = 0.5):
        self.store = store
        self.lam = float(lam)
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []

    def rebuild_bm25(self) -> None:
        all_ = self.store.get_all()
        self._ids = list(all_["ids"])
        self._texts = list(all_["documents"])
        tokenised = [_tokenise(t) for t in self._texts]
        if not tokenised:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(tokenised)

    def retrieve(
        self, query: str, query_embedding: np.ndarray, k: int = 5
    ) -> list[RetrievedChunk]:
        if self._bm25 is None:
            self.rebuild_bm25()
        dense_rows = self.store.query(query_embedding, k=max(k * 4, 16))
        dense_scores: dict[str, float] = {r["chunk_id"]: r["similarity"] for r in dense_rows}
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None and self._ids:
            tok = _tokenise(query)
            scores = self._bm25.get_scores(tok)
            if scores.max() > 0:
                scores = scores / scores.max()
            for cid, s in zip(self._ids, scores):
                bm25_scores[cid] = float(s)
        combined: dict[str, float] = {}
        for cid in set(list(dense_scores) + list(bm25_scores)):
            combined[cid] = self.lam * dense_scores.get(cid, 0.0) + (1 - self.lam) * bm25_scores.get(
                cid, 0.0
            )
        # top-k
        topk_ids = sorted(combined, key=combined.get, reverse=True)[:k]  # type: ignore[arg-type]
        rows_by_id = {r["chunk_id"]: r for r in dense_rows}
        out: list[RetrievedChunk] = []
        for cid in topk_ids:
            r = rows_by_id.get(cid)
            if r is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=r["text"],
                    metadata=r["metadata"],
                    similarity=combined[cid],
                    embedding=r.get("embedding"),
                )
            )
        return out
