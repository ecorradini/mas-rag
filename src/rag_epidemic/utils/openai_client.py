"""OpenAI client wrapper with on-disk caching and retry."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .hashing import stable_hash
from .paths import CACHE_DIR, get_openai_key
from .logging import get_logger

log = get_logger("openai-cache")


@dataclass
class UsageStats:
    completion_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def cost_usd(
        self,
        completion_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> float:
        # Prices in USD per 1M tokens (Nov 2024 pricing for gpt-4o-mini and emb-3-small)
        in_price = {"gpt-4o-mini": 0.150, "gpt-4o": 2.50}.get(completion_model, 0.150)
        out_price = {"gpt-4o-mini": 0.600, "gpt-4o": 10.0}.get(completion_model, 0.600)
        emb_price = {"text-embedding-3-small": 0.020, "text-embedding-3-large": 0.130}.get(
            embedding_model, 0.020
        )
        return (
            self.prompt_tokens * in_price
            + self.completion_tokens * out_price
            + self.embedding_tokens * emb_price
        ) / 1_000_000


_USAGE = UsageStats()


def usage() -> UsageStats:
    return _USAGE


class OpenAIClient:
    """Thin OpenAI wrapper with stable on-disk cache and global usage tracking."""

    def __init__(
        self,
        completion_model: str = "gpt-4o-mini-2024-07-18",
        embedding_model: str = "text-embedding-3-small",
        cache_dir: Path | None = None,
        enable_cache: bool = True,
    ):
        key = get_openai_key()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; create experiments/.env with the key."
            )
        self.client = OpenAI(api_key=key)
        self.completion_model = completion_model
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir or CACHE_DIR / "openai")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache

    # ---------- cache ----------
    def _cache_path(self, kind: str, key: str) -> Path:
        sub = self.cache_dir / kind / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def _cache_get(self, kind: str, key: str) -> Any | None:
        if not self.enable_cache:
            return None
        p = self._cache_path(kind, key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _cache_put(self, kind: str, key: str, value: Any) -> None:
        if not self.enable_cache:
            return
        p = self._cache_path(kind, key)
        p.write_text(json.dumps(value))

    # ---------- chat ----------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _chat_call(self, **kwargs: Any) -> Any:
        return self.client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format: dict | None = None,
        model: str | None = None,
    ) -> str:
        model = model or self.completion_model
        cache_key = stable_hash(
            dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                response_format=response_format,
            )
        )
        cached = self._cache_get("chat", cache_key)
        if cached is not None:
            _USAGE.completion_calls += 1
            _USAGE.prompt_tokens += cached.get("p_toks", 0)
            _USAGE.completion_tokens += cached.get("c_toks", 0)
            return cached["text"]

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        if response_format is not None:
            kwargs["response_format"] = response_format
        t0 = time.time()
        resp = self._chat_call(**kwargs)
        dt = time.time() - t0
        text = resp.choices[0].message.content or ""
        p_toks = resp.usage.prompt_tokens if resp.usage else 0
        c_toks = resp.usage.completion_tokens if resp.usage else 0
        _USAGE.completion_calls += 1
        _USAGE.prompt_tokens += p_toks
        _USAGE.completion_tokens += c_toks
        self._cache_put(
            "chat", cache_key, {"text": text, "p_toks": p_toks, "c_toks": c_toks, "dt": dt}
        )
        return text

    # ---------- embeddings ----------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _embed_call(self, **kwargs: Any) -> Any:
        return self.client.embeddings.create(**kwargs)

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or self.embedding_model
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        miss_texts: list[str] = []
        for i, t in enumerate(texts):
            key = stable_hash({"model": model, "text": t})
            cached = self._cache_get("emb", key)
            if cached is not None:
                _USAGE.embedding_calls += 1
                out[i] = cached["vec"]
            else:
                misses.append(i)
                miss_texts.append(t)
        if miss_texts:
            # batch up to 256 at a time
            for start in range(0, len(miss_texts), 256):
                batch = miss_texts[start : start + 256]
                resp = self._embed_call(model=model, input=batch)
                for off, item in enumerate(resp.data):
                    idx = misses[start + off]
                    vec = list(item.embedding)
                    out[idx] = vec
                    key = stable_hash({"model": model, "text": batch[off]})
                    self._cache_put("emb", key, {"vec": vec})
                _USAGE.embedding_tokens += resp.usage.total_tokens if resp.usage else 0
                _USAGE.embedding_calls += len(batch)
        return out  # type: ignore[return-value]
