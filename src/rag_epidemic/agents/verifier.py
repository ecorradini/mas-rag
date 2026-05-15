"""Epistemic Verifier (LLM-backed corroboration check)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..rag.retrievers import RetrievedChunk
from ..utils.openai_client import OpenAIClient

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_system() -> str:
    return (_PROMPT_DIR / "verifier.txt").read_text(encoding="utf-8")


@dataclass
class VerifierVerdict:
    verdict: str  # corroborated | contradicted | uncorroborated
    confidence: float
    rationale: str


class EpistemicVerifier:
    def __init__(self, client: OpenAIClient, m: int = 2, n: int = 3, seed: int = 0):
        self.client = client
        self.m = m
        self.n = n
        self.system = _load_system()
        self.seed = seed

    def verify(
        self, candidate: RetrievedChunk, corroborators: list[RetrievedChunk]
    ) -> VerifierVerdict:
        corr = "\n".join(
            f"[id={c.chunk_id} author={c.metadata.get('author_agent')} "
            f"channel={c.metadata.get('channel')} t={c.metadata.get('t')}] {c.text}"
            for c in corroborators[: self.n]
        )
        user = (
            f"CANDIDATE:\n[id={candidate.chunk_id} author={candidate.metadata.get('author_agent')} "
            f"channel={candidate.metadata.get('channel')} t={candidate.metadata.get('t')}] "
            f"{candidate.text}\n\nCORROBORATORS (up to n={self.n}):\n{corr}\n\n"
            f"Require at least m={self.m} agreeing sources. Reply with the JSON schema."
        )
        text = self.client.chat(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=120,
            seed=self.seed,
            response_format={"type": "json_object"},
        )
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
        return VerifierVerdict(
            verdict=str(obj.get("verdict", "uncorroborated")),
            confidence=float(obj.get("confidence", 0.0) or 0.0),
            rationale=str(obj.get("rationale", ""))[:200],
        )
