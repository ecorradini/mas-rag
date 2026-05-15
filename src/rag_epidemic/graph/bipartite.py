"""Dynamic bipartite agent–chunk graph built from the orchestrator JSONL log."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


@dataclass
class BipartiteGraph:
    """Bipartite multigraph: nodes have bipartite={0:'agent',1:'chunk'}.
    Edges carry t (superstep) and kind ('authored'|'retrieved')."""
    G: nx.MultiGraph = field(default_factory=nx.MultiGraph)

    def add_agent(self, agent_id: str) -> None:
        if agent_id not in self.G:
            self.G.add_node(agent_id, bipartite=0, kind="agent")

    def add_chunk(self, chunk_id: str, **attrs) -> None:
        if chunk_id not in self.G:
            self.G.add_node(chunk_id, bipartite=1, kind="chunk", **attrs)

    def authored(self, agent_id: str, chunk_id: str, t: int) -> None:
        self.add_agent(agent_id)
        self.add_chunk(chunk_id)
        self.G.add_edge(agent_id, chunk_id, t=t, kind="authored")

    def retrieved(self, agent_id: str, chunk_id: str, t: int) -> None:
        self.add_agent(agent_id)
        self.add_chunk(chunk_id)
        self.G.add_edge(agent_id, chunk_id, t=t, kind="retrieved")

    def agents(self) -> list[str]:
        return [n for n, d in self.G.nodes(data=True) if d.get("bipartite") == 0]

    def chunks(self) -> list[str]:
        return [n for n, d in self.G.nodes(data=True) if d.get("bipartite") == 1]

    def k_means(self) -> tuple[float, float]:
        agts = self.agents()
        chks = self.chunks()
        if not agts or not chks:
            return 0.0, 0.0
        return (
            sum(self.G.degree(a) for a in agts) / len(agts),
            sum(self.G.degree(c) for c in chks) / len(chks),
        )


def build_from_jsonl(path: str | Path) -> BipartiteGraph:
    bg = BipartiteGraph()
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            ev = json.loads(line)
            kind = ev.get("type")
            t = int(ev.get("t", 0))
            if kind == "chunk_written":
                bg.authored(ev["agent_id"], ev["chunk_id"], t)
                meta = ev.get("metadata") or {}
                bg.add_chunk(ev["chunk_id"], **{k: v for k, v in meta.items()
                                                if isinstance(v, (int, float, str, bool))})
            elif kind == "chunk_retrieved":
                bg.retrieved(ev["agent_id"], ev["chunk_id"], t)
    return bg
