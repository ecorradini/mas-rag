"""REALM-Bench-style logistics scenarios with deterministic scoring.

A *task* is a sequence of queries the operational agents must answer
correctly. Ground truth comes from the SimEngine, so scoring is
adversary-proof.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from ..corpus.ground_truth import SimEngine

Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class Question:
    qid: str
    role: Literal["demand", "inventory", "routing"]
    text: str
    warehouse: str | None
    correct_answer: str  # canonical string form
    correct_value: float | int | None


@dataclass
class TaskInstance:
    task_id: str
    difficulty: Difficulty
    seed: int
    engine: SimEngine
    questions_per_step: int = 3


def make_task(seed: int, difficulty: Difficulty = "medium", n_warehouses: int = 4) -> TaskInstance:
    eng = SimEngine(seed=seed, n_warehouses=n_warehouses, difficulty=difficulty)
    return TaskInstance(task_id=f"realm_logistics_{difficulty}_{seed}", difficulty=difficulty,
                        seed=seed, engine=eng)


def questions_for_step(task: TaskInstance) -> list[Question]:
    eng = task.engine
    rng = random.Random(hash((task.seed, eng.world.t)) & 0xFFFFFFFF)
    qs: list[Question] = []
    wh_names = list(eng.world.warehouses.keys())
    for i in range(task.questions_per_step):
        wh = rng.choice(wh_names)
        role = rng.choice(["demand", "inventory", "routing"])
        if role == "demand":
            text = f"What is the current forecast demand at warehouse {wh}?"
            val = eng.demand(wh)
            qs.append(Question(qid=f"q_{eng.world.t}_{i}", role="demand", text=text,
                               warehouse=wh, correct_answer=str(val), correct_value=val))
        elif role == "inventory":
            text = f"Is warehouse {wh} within its physical capacity at present?"
            within = eng.occupancy(wh) <= eng.capacity(wh)
            ans = "yes" if within else "no"
            qs.append(Question(qid=f"q_{eng.world.t}_{i}", role="inventory", text=text,
                               warehouse=wh, correct_answer=ans,
                               correct_value=1.0 if within else 0.0))
        else:
            text = f"How many units of capacity does warehouse {wh} have available?"
            avail = max(0, eng.capacity(wh) - eng.occupancy(wh))
            qs.append(Question(qid=f"q_{eng.world.t}_{i}", role="routing", text=text,
                               warehouse=wh, correct_answer=str(avail), correct_value=avail))
    return qs


def score_answer(q: Question, agent_answer_value: float | int | str | None,
                 tol: float = 0.10) -> bool:
    """Return True iff the agent's answer is within tolerance of ground truth."""
    if agent_answer_value is None:
        return False
    if q.correct_value is None:
        return str(agent_answer_value).strip().lower() == str(q.correct_answer).strip().lower()
    try:
        v = float(agent_answer_value)
    except (TypeError, ValueError):
        try:
            # numeric inside string
            v = float(str(agent_answer_value).split()[0].replace(",", ""))
        except Exception:
            return False
    gt = float(q.correct_value)
    if gt == 0:
        return abs(v - gt) <= max(1.0, tol)
    return abs(v - gt) / max(abs(gt), 1.0) <= tol
