"""Deterministic SimEvent log — the ground truth that wraps every chunk.

OrgForge's core insight: separate the deterministic Python engine from the
LLM prose. We implement a self-contained logistics SimEvent generator that
produces a stream of events (warehouse capacities, vehicle assignments,
incidents) at integer supersteps. Each event has a stable ID; chunks
generated from it inherit `sim_event_id`.

The ground truth is therefore knowable without any LLM call.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

EventKind = Literal[
    "warehouse_capacity", "vehicle_assignment", "incident", "demand_forecast",
    "route_completed",
]


@dataclass
class SimEvent:
    event_id: str
    t: int
    kind: EventKind
    payload: dict
    channel: str  # slack | jira | confluence | email
    is_ground_truth: bool = True  # False ⇒ injected by Patient Zero
    extra: dict = field(default_factory=dict)


# ---------- domain ----------
WAREHOUSES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
VEHICLES = [f"V{n:03d}" for n in range(1, 21)]
CHANNELS = ["slack", "jira", "confluence", "email"]


@dataclass
class WarehouseState:
    name: str
    capacity: int          # ground truth, immutable
    occupancy: int = 0     # current
    incidents: int = 0


@dataclass
class WorldState:
    t: int = 0
    warehouses: dict[str, WarehouseState] = field(default_factory=dict)
    demand_by_warehouse: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "warehouses": {n: asdict(w) for n, w in self.warehouses.items()},
            "demand_by_warehouse": dict(self.demand_by_warehouse),
        }


# ---------- engine ----------
class SimEngine:
    """Deterministic logistics simulator. All randomness from a seeded RNG."""

    def __init__(
        self,
        seed: int,
        n_warehouses: int = 4,
        difficulty: str = "medium",
        base_capacity: tuple[int, int] = (300, 600),
    ):
        self.rng = random.Random(seed)
        names = WAREHOUSES[:n_warehouses]
        self.world = WorldState(
            warehouses={
                n: WarehouseState(name=n, capacity=self.rng.randint(*base_capacity))
                for n in names
            }
        )
        self.difficulty = difficulty
        self._eid_counter = 0
        # difficulty knobs
        self.demand_noise = {"easy": 0.05, "medium": 0.15, "hard": 0.35}[difficulty]
        self.incident_p = {"easy": 0.0, "medium": 0.03, "hard": 0.08}[difficulty]
        self.disruption_p = {"easy": 0.0, "medium": 0.05, "hard": 0.15}[difficulty]

    def _next_eid(self) -> str:
        self._eid_counter += 1
        return f"evt_{self._eid_counter:06d}"

    # ---- public ----
    def step(self) -> list[SimEvent]:
        self.world.t += 1
        events: list[SimEvent] = []
        # demand forecast
        for name, w in self.world.warehouses.items():
            base = int(w.capacity * 0.5)
            noise = int(base * self.demand_noise * (self.rng.random() * 2 - 1))
            demand = max(0, base + noise)
            self.world.demand_by_warehouse[name] = demand
            events.append(
                SimEvent(
                    event_id=self._next_eid(),
                    t=self.world.t,
                    kind="demand_forecast",
                    payload={"warehouse": name, "demand": demand, "capacity": w.capacity},
                    channel="slack",
                )
            )
            # update occupancy
            w.occupancy = min(w.capacity, demand)
        # vehicle assignments
        for v in self.rng.sample(VEHICLES, k=min(4, len(VEHICLES))):
            wh = self.rng.choice(list(self.world.warehouses.keys()))
            events.append(
                SimEvent(
                    event_id=self._next_eid(),
                    t=self.world.t,
                    kind="vehicle_assignment",
                    payload={"vehicle": v, "warehouse": wh},
                    channel="email",
                )
            )
        # incidents
        for name, w in self.world.warehouses.items():
            if self.rng.random() < self.incident_p:
                w.incidents += 1
                events.append(
                    SimEvent(
                        event_id=self._next_eid(),
                        t=self.world.t,
                        kind="incident",
                        payload={"warehouse": name, "severity": self.rng.randint(1, 3)},
                        channel="jira",
                    )
                )
        # rare disruption: capacity change is *never* injected by the engine
        # (capacities are physical and immutable); only Patient Zero will lie
        # about capacity.
        return events

    # ---- queries (ground truth) ----
    def capacity(self, warehouse: str) -> int:
        return self.world.warehouses[warehouse].capacity

    def occupancy(self, warehouse: str) -> int:
        return self.world.warehouses[warehouse].occupancy

    def demand(self, warehouse: str) -> int:
        return self.world.demand_by_warehouse.get(warehouse, 0)


def dump_events(events: list[SimEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for e in events:
            fp.write(json.dumps(asdict(e)) + "\n")
