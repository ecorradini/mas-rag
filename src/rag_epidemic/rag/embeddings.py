"""Embedding interface — OpenAI primary, optional local fallback."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..utils.openai_client import OpenAIClient


class Embedder(Protocol):
    dim: int
    name: str
    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", client: OpenAIClient | None = None):
        self.client = client or OpenAIClient(embedding_model=model)
        self.model = model
        self.name = model
        self.dim = 1536 if "small" in model else 3072

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.client.embed(texts, model=self.model)
        return np.asarray(vecs, dtype=np.float32)


class LocalSTEmbedder:
    """sentence-transformers based local embedder (for ablations)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def build_embedder(spec: str) -> Embedder:
    if spec.startswith("openai:"):
        return OpenAIEmbedder(model=spec.split(":", 1)[1])
    if spec.startswith("local:"):
        return LocalSTEmbedder(model_name=spec.split(":", 1)[1])
    raise ValueError(f"Unknown embedder spec: {spec}")
