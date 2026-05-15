"""Healthy LLM-backed operational agents (Demand, Inventory, Routing)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..rag.retrievers import RetrievedChunk
from ..utils.openai_client import OpenAIClient

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_system(name: str) -> str:
    return (_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        m = c.metadata
        lines.append(
            f"[id={c.chunk_id} author={m.get('author_agent')} channel={m.get('channel')} "
            f"t={m.get('t')} sim={m.get('sim_event_id')}] {c.text}"
        )
    return "\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"answer": None, "rationale": "parse_error", "evidence_chunk_ids": []}


@dataclass
class AgentAnswer:
    agent_id: str
    role: str
    question_id: str
    answer: Any
    rationale: str
    evidence_chunk_ids: list[str]
    raw: str


class HealthyAgent:
    """LLM-backed agent with a single operational role."""

    def __init__(
        self,
        agent_id: str,
        role: str,  # demand | inventory | routing
        client: OpenAIClient,
        seed: int = 0,
    ):
        self.agent_id = agent_id
        self.role = role
        self.client = client
        self.system_prompt = _load_system(role)
        self.seed = seed

    def answer(
        self, question_text: str, question_id: str, context: list[RetrievedChunk]
    ) -> AgentAnswer:
        user = (
            "CONTEXT (retrieved chunks):\n"
            + _format_context(context)
            + f"\n\nQUESTION: {question_text}\n\nReply with the JSON schema."
        )
        text = self.client.chat(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=200,
            seed=self.seed,
            response_format={"type": "json_object"},
        )
        obj = _parse_json(text)
        return AgentAnswer(
            agent_id=self.agent_id,
            role=self.role,
            question_id=question_id,
            answer=obj.get("answer"),
            rationale=str(obj.get("rationale", ""))[:200],
            evidence_chunk_ids=list(obj.get("evidence_chunk_ids") or []),
            raw=text,
        )

    def write_observation(
        self, question_text: str, ans: AgentAnswer
    ) -> str:
        """Produce a short channel-style chunk that the agent will publish."""
        msg = self.client.chat(
            messages=[
                {"role": "system", "content": (
                    "You are a logistics agent that just produced an answer. "
                    "Write ONE concise Slack-style status update (max 25 words) "
                    "stating the answer, in third person, no JSON, no preamble.")},
                {"role": "user", "content": (
                    f"Question: {question_text}\nAnswer: {ans.answer}\n"
                    f"Rationale: {ans.rationale}"
                )},
            ],
            temperature=0.3,
            max_tokens=80,
            seed=self.seed + 1,
        )
        return msg.strip()
