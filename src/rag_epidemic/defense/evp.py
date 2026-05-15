"""EVP (Epistemic Verifier Patrol) — our proposed defense.

Combines:
  1. Centrality-targeted patrol: top-k* chunks by dual centrality are
     re-verified by the Epistemic Verifier (m-of-n corroboration).
  2. Trusted-baseline corroborators: corroborators are drawn preferentially
     from the oldest available system-authored chunks (the immutable
     ground-truth backbone), not from co-retrieved (possibly poisoned)
     candidates.
  3. Quarantine: chunks marked 'contradicted' (or 'uncorroborated' when
     ``quarantine_on_uncorroborated``) are excluded from retrieval.
  4. Trust feedback + permanent blacklist: M-ASK-style score for the author
     updated on each verdict; after `blacklist_after` contradictions an
     author is blacklisted indefinitely.

Implements Algorithm 1 of the paper.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..agents.verifier import EpistemicVerifier
from ..graph.bipartite import BipartiteGraph
from ..graph.centrality import dual_centrality
from ..rag.retrievers import RetrievedChunk


class EVP:
    name = "evp"

    def __init__(
        self,
        verifier: EpistemicVerifier,
        kstar_frac: float = 0.10,
        gamma: float = 0.6,
        trust_init: float = 1.0,
        trust_threshold: float = 0.5,
        trust_decay: float = 0.35,
        m: int = 2,
        n: int = 3,
        quarantine_on_uncorroborated: bool = True,
        blacklist_after: int = 2,
        trusted_authors: tuple[str, ...] = ("system",),
        ingest_sim_threshold: float = 0.55,
    ):
        self.v = verifier
        self.v.m = m
        self.v.n = n
        self.kstar_frac = kstar_frac
        self.gamma = gamma
        self.trust: dict[str, float] = defaultdict(lambda: trust_init)
        self.trust_threshold = trust_threshold
        self.trust_decay = trust_decay
        self.trust_init = trust_init
        self.quarantine: set[str] = set()
        self.uncorroborated_quarantine = quarantine_on_uncorroborated
        self.blacklist: set[str] = set()
        self.blacklist_after = blacklist_after
        self._contradictions: dict[str, int] = defaultdict(int)
        self.trusted_authors = set(trusted_authors)
        self.ingest_sim_threshold = ingest_sim_threshold
        self._graph: BipartiteGraph | None = None
        self._all_candidates_cache: list[RetrievedChunk] = []
        self._trusted_pool: list[RetrievedChunk] = []  # backbone for corroboration
        self._max_trusted_pool = 200

    # ---------------- public hooks ----------------
    def set_graph(self, bg: BipartiteGraph) -> None:
        self._graph = bg

    def stage_corroborators(self, pool: list[RetrievedChunk]) -> None:
        self._all_candidates_cache = pool
        # also seed the trusted pool with any system-authored chunks we have seen
        for c in pool:
            a = c.metadata.get("author_agent", "?")
            if a in self.trusted_authors and c.chunk_id not in {x.chunk_id for x in self._trusted_pool}:
                self._trusted_pool.append(c)
        # keep oldest first, capped
        self._trusted_pool.sort(key=lambda c: c.metadata.get("t", 0))
        self._trusted_pool = self._trusted_pool[: self._max_trusted_pool]

    def filter(self, query, query_embedding, candidates: list[RetrievedChunk]):
        # 0) drop blacklisted authors outright
        cand = [c for c in candidates
                if c.metadata.get("author_agent") not in self.blacklist]
        # 1) drop already-quarantined chunks & low-trust authors
        cand = [c for c in cand
                if c.chunk_id not in self.quarantine and
                self.trust[c.metadata.get("author_agent", "?")] >= self.trust_threshold]
        if not cand:
            # fall back to trusted-author chunks if everything is suspicious
            cand = [c for c in candidates
                    if c.metadata.get("author_agent") in self.trusted_authors
                    and c.chunk_id not in self.quarantine]
        if not cand:
            return candidates[:1]

        # 2) score by dual centrality (if graph available); else by similarity
        if self._graph is not None:
            cent = dual_centrality(self._graph, gamma=self.gamma)
            scored = sorted(cand, key=lambda c: cent.get(c.chunk_id, 0.0), reverse=True)
        else:
            scored = sorted(cand, key=lambda c: c.similarity, reverse=True)
        k_star = max(1, int(np.ceil(self.kstar_frac * len(scored))))
        patrol = scored[:k_star]
        # also always patrol any non-trusted authors in the top-k
        for c in scored[: max(3, k_star)]:
            if c.metadata.get("author_agent") not in self.trusted_authors and c not in patrol:
                patrol.append(c)

        # 3) verify patrol set against TRUSTED corroborators
        for cand_chunk in patrol:
            if cand_chunk.metadata.get("author_agent") in self.trusted_authors:
                continue  # don't verify backbone against itself
            corrs = self._pick_corroborators(cand_chunk, k=self.v.n)
            if not corrs:
                continue
            # fast numeric short-circuit
            if self._numeric_contradicts(cand_chunk, corrs):
                self.quarantine.add(cand_chunk.chunk_id)
                author = cand_chunk.metadata.get("author_agent", "?")
                self._contradictions[author] += 1
                self._update_trust(author, "contradicted")
                if self._contradictions[author] >= self.blacklist_after:
                    self.blacklist.add(author)
                continue
            verdict = self.v.verify(cand_chunk, corrs)
            author = cand_chunk.metadata.get("author_agent", "?")
            self._update_trust(author, verdict.verdict)
            if verdict.verdict == "contradicted":
                self.quarantine.add(cand_chunk.chunk_id)
                self._contradictions[author] += 1
                if self._contradictions[author] >= self.blacklist_after:
                    self.blacklist.add(author)
            elif verdict.verdict == "uncorroborated" and self.uncorroborated_quarantine:
                self.quarantine.add(cand_chunk.chunk_id)

        # 4) return filtered set
        out = [c for c in cand
               if c.chunk_id not in self.quarantine
               and c.metadata.get("author_agent") not in self.blacklist
               and self.trust[c.metadata.get("author_agent", "?")] >= self.trust_threshold]
        if not out:
            out = [c for c in candidates
                   if c.metadata.get("author_agent") in self.trusted_authors
                   and c.chunk_id not in self.quarantine][:1]
        return out or cand[:1]

    def _numeric_claim(self, text: str) -> tuple[str, float] | None:
        """Extract (warehouse_id, numeric_value) from a chunk's text if any."""
        import re
        m_wh = re.search(r"\b(WH[0-9A-Za-z_-]+|warehouse\s+[A-Za-z0-9_-]+)", text)
        if not m_wh:
            return None
        wh = m_wh.group(0).split()[-1]
        nums = re.findall(r"(?<![\d.])\d{2,6}(?:\.\d+)?(?![\d.])", text)
        if not nums:
            return None
        try:
            v = float(nums[0])
        except ValueError:
            return None
        return wh, v

    def _numeric_contradicts(self, c: RetrievedChunk,
                             corrs: list[RetrievedChunk]) -> bool:
        cand = self._numeric_claim(c.text)
        if not cand:
            return False
        wh_c, v_c = cand
        # only check capacity-flavoured candidates (the PZ attack vector)
        if not any(w in c.text.lower()
                   for w in ("capacit", "available", "max load", "max-load", "throughput")):
            return False
        for d in corrs:
            ref = self._numeric_claim(d.text)
            if not ref:
                continue
            wh_d, v_d = ref
            if wh_d != wh_c:
                continue
            if not any(w in d.text.lower()
                       for w in ("capacit", "available", "max load", "max-load", "throughput")):
                continue
            if v_d <= 0:
                continue
            ratio = v_c / v_d
            if ratio >= 1.25 or ratio <= 0.75:
                return True
        return False

    def pre_ingest_inspect(self, new_chunks: list[RetrievedChunk]) -> set[str]:
        """Inspect freshly-authored chunks before they enter the store.

        Only blocks when (a) the candidate is on the same TOPIC as a trusted
        corroborator (cosine sim ≥ ``ingest_sim_threshold``) and (b) the
        verifier returns 'contradicted'. This avoids rejecting legitimate
        observations whose subject simply isn't covered by the backbone.
        """
        blocked: set[str] = set()
        for c in new_chunks:
            author = c.metadata.get("author_agent", "?")
            if author in self.trusted_authors:
                continue
            if author in self.blacklist:
                blocked.add(c.chunk_id)
                continue
            corrs = self._pick_corroborators(c, k=self.v.n)
            if not corrs:
                continue
            # topical-similarity gate: skip if no on-topic corroborator
            if c.embedding is not None:
                max_sim = 0.0
                for d in corrs:
                    if d.embedding is None:
                        continue
                    s = float(np.dot(c.embedding, d.embedding) /
                              (np.linalg.norm(c.embedding) * np.linalg.norm(d.embedding) + 1e-8))
                    if s > max_sim:
                        max_sim = s
                if max_sim < self.ingest_sim_threshold:
                    continue
            # Fast numeric sanity-check: catch capacity inflation without LLM
            if self._numeric_contradicts(c, corrs):
                blocked.add(c.chunk_id)
                self.quarantine.add(c.chunk_id)
                self._contradictions[author] += 1
                if self._contradictions[author] >= self.blacklist_after:
                    self.blacklist.add(author)
                continue
            verdict = self.v.verify(c, corrs)
            self._update_trust(author, verdict.verdict)
            if verdict.verdict == "contradicted":
                blocked.add(c.chunk_id)
                self.quarantine.add(c.chunk_id)
                self._contradictions[author] += 1
                if self._contradictions[author] >= self.blacklist_after:
                    self.blacklist.add(author)
        return blocked

    def update(self, **kwargs):
        ev = kwargs.get("event")
        if not ev:
            return

    # ---------------- helpers ----------------
    def _pick_corroborators(self, c: RetrievedChunk, k: int) -> list[RetrievedChunk]:
        # Prefer trusted (system-authored) pool, oldest first
        trusted = [d for d in self._trusted_pool
                   if d.chunk_id != c.chunk_id and d.chunk_id not in self.quarantine]
        if c.embedding is not None and trusted:
            def sim(d):
                if d.embedding is None:
                    return 0.0
                return float(np.dot(c.embedding, d.embedding) /
                             (np.linalg.norm(c.embedding) * np.linalg.norm(d.embedding) + 1e-8))
            trusted = sorted(trusted, key=sim, reverse=True)
        if len(trusted) >= k:
            return trusted[:k]
        # fall back to candidates pool for the remainder
        pool = [d for d in self._all_candidates_cache
                if d.chunk_id != c.chunk_id and
                d.metadata.get("author_agent") != c.metadata.get("author_agent") and
                d.metadata.get("author_agent") not in self.blacklist and
                d.chunk_id not in self.quarantine]
        if c.embedding is not None:
            def sim2(d):
                if d.embedding is None:
                    return 0.0
                return float(np.dot(c.embedding, d.embedding) /
                             (np.linalg.norm(c.embedding) * np.linalg.norm(d.embedding) + 1e-8))
            pool = sorted(pool, key=sim2, reverse=True)
        return (trusted + pool)[:k]

    def _update_trust(self, author: str, verdict: str) -> None:
        if author in self.trusted_authors:
            return
        if verdict == "contradicted":
            self.trust[author] = max(0.0, self.trust[author] - self.trust_decay)
        elif verdict == "corroborated":
            self.trust[author] = min(self.trust_init, self.trust[author] + self.trust_decay * 0.3)

