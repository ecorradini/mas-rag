"""Scoring utilities for REALM-Bench-style tasks."""

from __future__ import annotations

from .realm_bench import Question, score_answer


def episode_accuracy(qs: list[Question], answers: dict[str, str | float | int | None]) -> float:
    if not qs:
        return 1.0
    correct = sum(1 for q in qs if score_answer(q, answers.get(q.qid)))
    return correct / len(qs)
