"""Dual centrality: γ·betweenness + (1-γ)·normalised degree."""

from __future__ import annotations

import networkx as nx

from .bipartite import BipartiteGraph


def dual_centrality(bg: BipartiteGraph, gamma: float = 0.6) -> dict[str, float]:
    G = bg.G
    if len(G) == 0:
        return {}
    # Project to one-mode for betweenness (use a simple unprojected version
    # but limit to chunk nodes since that is the surface adversary controls).
    chunks = bg.chunks()
    if not chunks:
        return {}
    sub = nx.Graph()
    sub.add_nodes_from(chunks)
    # connect chunks that share an agent (co-retrieval / co-authorship)
    by_agent: dict[str, list[str]] = {}
    for u, v, _ in G.edges(data=True):
        a, c = (u, v) if G.nodes[u]["bipartite"] == 0 else (v, u)
        by_agent.setdefault(a, []).append(c)
    for cs in by_agent.values():
        cs = list(set(cs))
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                sub.add_edge(cs[i], cs[j])
    btw = nx.betweenness_centrality(sub) if sub.number_of_nodes() > 1 else {c: 0.0 for c in chunks}
    degs = {c: G.degree(c) for c in chunks}
    max_d = max(degs.values()) if degs else 1
    if max_d == 0:
        max_d = 1
    return {c: gamma * btw.get(c, 0.0) + (1 - gamma) * (degs[c] / max_d) for c in chunks}
