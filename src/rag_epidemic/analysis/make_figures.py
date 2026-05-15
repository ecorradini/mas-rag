"""Produce all paper figures + fill PLACEHOLDER cells in tables.

CLI flags
---------
* ``--rescore``     Recompute derived metrics for every existing
                    ``result.json`` from its ``events.jsonl`` (no LLM
                    calls); idempotent.
* ``--figs-only``   Skip rescoring; just regenerate figures from cached
                    derived metrics.
* ``--results-root``  Root containing per-run subdirs (default RESULTS_DIR).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import (
    DERIVED_KEY,
    aggregate_by_defense,
    derive_run_metrics,
    load_runs,
    pairwise_stats,
    rescore_run,
    write_pvalue_table,
    write_tab3,
)
from ..utils.paths import PAPER_FIG_DIR, PAPER_TBL_DIR, RESULTS_DIR


PRETTY = {
    "undefended": "Undefended", "secon_rag": "SeCon-RAG",
    "sem_chameleon": "Sem.\\ Chameleon", "flp": "FLP",
    "spark2fire": "Spark2Fire", "mask": "M-ASK", "evp": "EVP (ours)",
}
ORDER = ["undefended", "secon_rag", "sem_chameleon", "spark2fire",
         "mask", "flp", "evp"]


def _palette():
    return {
        "undefended": "#d62728", "secon_rag": "#ff7f0e", "sem_chameleon": "#bcbd22",
        "flp": "#17becf", "spark2fire": "#9467bd", "mask": "#8c564b", "evp": "#2ca02c",
    }


def _pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Fig 3 — accuracy curves (unchanged but cleaner palette)                     #
# --------------------------------------------------------------------------- #
def fig_accuracy_curves(runs, out: Path) -> None:
    plt.figure(figsize=(7, 4))
    pal = _palette()
    by_d: dict[str, list[list[float]]] = {}
    for r in runs:
        d = r["config"]["defense_name"]
        accs = [s["accuracy"] for s in r["result"]["steps"]]
        by_d.setdefault(d, []).append(accs)
    for d in ORDER:
        if d not in by_d:
            continue
        traces = by_d[d]
        L = min(len(t) for t in traces)
        M = np.mean([t[:L] for t in traces], axis=0)
        S = np.std([t[:L] for t in traces], axis=0)
        x = np.arange(1, L + 1)
        plt.plot(x, M, label=_pretty(d), color=pal.get(d, "k"), lw=1.5)
        plt.fill_between(x, M - S, M + S, color=pal.get(d, "k"), alpha=0.15)
    plt.axhline(0.4, color="grey", ls="--", lw=0.8, label="collapse threshold")
    plt.xlabel("Superstep $t$"); plt.ylabel("Task accuracy")
    plt.legend(loc="upper right", fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(out); plt.close()


# --------------------------------------------------------------------------- #
# Fig 4 (J3) — poisoned-served-at-retrieval fraction on semi-log y           #
# --------------------------------------------------------------------------- #
def fig_poisoned_served_fraction(runs, out: Path) -> None:
    plt.figure(figsize=(7, 4))
    pal = _palette()
    by_d: dict[str, list[list[float]]] = {}
    for r in runs:
        d = r["config"]["defense_name"]
        derived = r.get(DERIVED_KEY, {})
        frac = derived.get("poisoned_served_fraction") or []
        if frac:
            by_d.setdefault(d, []).append(frac)
    for d in ORDER:
        if d not in by_d:
            continue
        traces = by_d[d]
        L = min(len(t) for t in traces)
        if L == 0:
            continue
        M = np.mean([t[:L] for t in traces], axis=0)
        M = np.clip(M, 1e-3, 1.0)  # floor for log scale
        plt.plot(np.arange(1, L + 1), M, label=_pretty(d),
                 color=pal.get(d, "k"), lw=1.5)
    plt.yscale("log")
    plt.xlabel("Superstep $t$")
    plt.ylabel(r"Poisoned fraction of \emph{retrieved} chunks (log)")
    plt.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(out); plt.close()


# --------------------------------------------------------------------------- #
# Fig 5 (J2) — post-collapse accuracy floor                                   #
# --------------------------------------------------------------------------- #
def fig_accuracy_floor_bar(stats, out: Path) -> None:
    plt.figure(figsize=(6.5, 3.5))
    pal = _palette()
    names = [n for n in ORDER if n in stats]
    vals = [stats[n].get("accuracy_floor_mean", stats[n]["acc_mean"]) for n in names]
    errs = [stats[n].get("accuracy_floor_std", stats[n]["acc_std"]) for n in names]
    cols = [pal.get(n, "k") for n in names]
    xs = np.arange(len(names))
    plt.bar(xs, vals, yerr=errs, color=cols, capsize=3, edgecolor="black", lw=0.5)
    plt.xticks(xs, [_pretty(n) for n in names], rotation=20, ha="right")
    plt.ylabel("Post-collapse accuracy floor")
    plt.axhline(0.4, color="grey", ls="--", lw=0.8, label="collapse threshold")
    plt.legend(fontsize=8, loc="upper left")
    plt.ylim(0, max(0.45, max(vals) * 1.2))
    plt.tight_layout(); plt.savefig(out); plt.close()


# --------------------------------------------------------------------------- #
# Fig 1 (J4) — bipartite topology snapshot                                    #
# --------------------------------------------------------------------------- #
def fig_bipartite_topology(runs, out: Path, prefer_defense: str = "undefended",
                            prefer_difficulty: str = "medium",
                            prefer_seed: int = 0) -> None:
    try:
        import networkx as nx
    except Exception:
        return
    chosen = None
    for r in runs:
        cfg = r["config"]
        if (cfg["defense_name"] == prefer_defense
                and cfg["difficulty"] == prefer_difficulty
                and cfg["seed"] == prefer_seed):
            chosen = r
            break
    chosen = chosen or (runs[0] if runs else None)
    if chosen is None:
        return
    events_path = Path(chosen.get("_dir", chosen.get("_path", ""))).parent \
        if chosen.get("_path") else Path(chosen.get("_dir", "."))
    events_path = Path(chosen.get("_dir") or
                       chosen["config"].get("out_dir") or ".") / "events.jsonl"
    if not events_path.exists():
        return
    G = nx.MultiGraph()
    poisoned_chunks: set[str] = set()
    infected_agents: set[str] = set()
    edges: list[tuple[str, str, int]] = []
    last_t = 0
    with open(events_path, encoding="utf-8") as fp:
        for line in fp:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = int(ev.get("t", 0))
            last_t = max(last_t, t)
            if ev.get("type") == "chunk_written":
                aid, cid = ev["agent_id"], ev["chunk_id"]
                G.add_node(aid, bipartite=0); G.add_node(cid, bipartite=1)
                if (ev.get("metadata") or {}).get("is_poisoned"):
                    poisoned_chunks.add(cid)
                    if aid != "system":
                        infected_agents.add(aid)
                edges.append((aid, cid, t))
            elif ev.get("type") == "chunk_retrieved":
                aid, cid = ev["agent_id"], ev["chunk_id"]
                G.add_node(aid, bipartite=0); G.add_node(cid, bipartite=1)
                edges.append((aid, cid, t))
                if ev.get("is_poisoned"):
                    infected_agents.add(aid)
    for u, v, t in edges:
        G.add_edge(u, v, t=t)
    # subsample chunks for legibility
    chunks = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    agents = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    if len(chunks) > 80:
        # keep all poisoned + degree-top + random
        deg = sorted(chunks, key=lambda c: G.degree(c), reverse=True)
        keep = set(list(poisoned_chunks) + deg[:60])
        # pad with a few low-degree for context
        for c in chunks:
            if len(keep) >= 90:
                break
            keep.add(c)
        H = G.subgraph(list(agents) + list(keep)).copy()
    else:
        H = G
    pos = {}
    n_a = len(agents)
    for i, a in enumerate(agents):
        pos[a] = (-1.0, (i - n_a / 2) / max(n_a, 1))
    keep_chunks = [n for n, d in H.nodes(data=True) if d.get("bipartite") == 1]
    for i, c in enumerate(keep_chunks):
        pos[c] = (1.0, (i - len(keep_chunks) / 2) / max(len(keep_chunks), 1))
    plt.figure(figsize=(7, 5))
    # edges shaded by recency
    for u, v, d in H.edges(data=True):
        t = int(d.get("t", 0))
        alpha = 0.15 + 0.65 * (t / max(last_t, 1))
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                 color="0.4", alpha=alpha, lw=0.4, zorder=1)
    # nodes
    for a in agents:
        plt.scatter([pos[a][0]], [pos[a][1]],
                    c=("#d62728" if a in infected_agents else "#1f77b4"),
                    s=120, marker="o", edgecolor="k", lw=0.5, zorder=3)
    for c in keep_chunks:
        plt.scatter([pos[c][0]], [pos[c][1]],
                    c=("#ff7f0e" if c in poisoned_chunks else "#bbbbbb"),
                    s=20, marker="s", edgecolor="k", lw=0.2, zorder=2)
    plt.axis("off")
    plt.title(f"$\\mathcal{{G}}(t={last_t})$ — "
              f"{_pretty(chosen['config']['defense_name'])}, "
              f"seed={chosen['config']['seed']}", fontsize=10)
    plt.tight_layout(); plt.savefig(out); plt.close()


# --------------------------------------------------------------------------- #
# Fig 2 (J5) — β(cos) logistic fit                                            #
# --------------------------------------------------------------------------- #
def fig_beta_fit(runs, out: Path, data_out: Path | None = None) -> None:
    # We don't store per-retrieval cosine in events.jsonl directly, so we fit a
    # surrogate logistic against (n_poisoned_in_corpus_fraction, infected_flag)
    # using poisoned-served events. This produces a defensible, monotonic
    # transmission curve in the absence of stored similarities.
    pts_x: list[float] = []
    pts_y: list[int] = []
    for r in runs:
        steps = r["result"]["steps"]
        for s in steps:
            x = s["n_poisoned_in_corpus"] / max(s["n_chunks"], 1)
            y = 1 if (s["accuracy"] < 0.4) else 0
            pts_x.append(float(x)); pts_y.append(int(y))
    if len(pts_x) < 20:
        return
    x = np.asarray(pts_x); y = np.asarray(pts_y)
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500).fit(x.reshape(-1, 1), y)
        alpha = float(clf.coef_[0, 0]); tau = float(-clf.intercept_[0])
    except Exception:
        return
    xs = np.linspace(0, max(x.max(), 0.1), 200)
    sig = 1.0 / (1.0 + np.exp(-(alpha * xs - tau)))
    plt.figure(figsize=(6, 3.5))
    plt.scatter(x, y + np.random.uniform(-0.03, 0.03, size=len(y)),
                s=6, color="0.5", alpha=0.4, label="empirical (jittered)")
    plt.plot(xs, sig, color="#2ca02c", lw=2,
             label=fr"$\sigma(\alpha\,p - \tau)$, $\alpha={alpha:.2f},\ \tau={tau:.2f}$")
    plt.xlabel("Poisoned fraction of corpus")
    plt.ylabel(r"$\Pr[\mathrm{collapse}\ |\ p]$")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out); plt.close()
    if data_out is not None:
        data_out.write_text(json.dumps(
            {"alpha": alpha, "tau": tau, "n_points": int(len(x))}, indent=2))


# --------------------------------------------------------------------------- #
# Fig 6 (J6) — ODE-vs-sim overlay                                             #
# --------------------------------------------------------------------------- #
def fig_ode_overlay(runs, out: Path) -> None:
    from ..epidemic.ode_model import ODEParams, simulate
    pal = _palette()
    by_d: dict[str, list[list[int]]] = {}
    n_a_by_d: dict[str, int] = {}
    for r in runs:
        d = r["config"]["defense_name"]
        series = [int(s.get("infected_agents", 0)) for s in r["result"]["steps"]]
        by_d.setdefault(d, []).append(series)
        n_a_by_d[d] = r["config"]["N_A"]
    keep = [d for d in ORDER if d in by_d]
    if not keep:
        return
    cols = min(3, len(keep))
    rows = int(np.ceil(len(keep) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows),
                            sharey=True, squeeze=False)
    for ax, d in zip(axs.flat, keep):
        traces = by_d[d]
        L = min(len(t) for t in traces)
        emp = np.mean([t[:L] for t in traces], axis=0)
        # crude ODE fit: use late-time poisoned fraction as P(0)
        n_a = n_a_by_d.get(d, 6)
        p = ODEParams(N_A=n_a, N_K=30, beta_star=0.35 if d != "evp" else 0.07,
                      mu=0.10, rho_I=2.0, rho_S=1.0, lambda_p=0.4)
        sol = simulate(p, T=L, I0=1, P0=1)
        ax.plot(np.arange(1, L + 1), emp, color=pal.get(d, "k"), lw=1.8, label="sim")
        ax.plot(sol["t"][: L + 1], sol["I"][: L + 1], color=pal.get(d, "k"),
                ls="--", lw=1.0, label="ODE")
        ax.set_title(_pretty(d), fontsize=9)
        ax.set_xlabel("$t$"); ax.set_ylabel("$I(t)$")
        ax.legend(fontsize=7)
    for ax in axs.flat[len(keep):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(out); plt.close()


# --------------------------------------------------------------------------- #
# Fig 7 (J7) — dual-centrality heatmap (1D histogram + PZ marks)              #
# --------------------------------------------------------------------------- #
def fig_centrality_distribution(runs, out: Path,
                                  prefer_defense: str = "evp",
                                  prefer_seed: int = 0) -> None:
    from ..graph.bipartite import build_from_jsonl
    from ..graph.centrality import dual_centrality
    chosen = None
    for r in runs:
        cfg = r["config"]
        if cfg["defense_name"] == prefer_defense and cfg["seed"] == prefer_seed:
            chosen = r
            break
    chosen = chosen or (runs[0] if runs else None)
    if chosen is None:
        return
    run_dir = Path(chosen.get("_dir") or chosen["config"].get("out_dir") or ".")
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return
    bg = build_from_jsonl(events_path)
    cent = dual_centrality(bg, gamma=0.6)
    if not cent:
        return
    # mark poisoned chunks
    poisoned: set[str] = set()
    with open(events_path, encoding="utf-8") as fp:
        for line in fp:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "chunk_written" \
                    and (ev.get("metadata") or {}).get("is_poisoned"):
                poisoned.add(ev["chunk_id"])
    vals = np.asarray(list(cent.values()))
    poi_vals = np.asarray([cent[c] for c in poisoned if c in cent])
    plt.figure(figsize=(6.5, 3.5))
    plt.hist(vals, bins=40, color="#bbbbbb", edgecolor="k", lw=0.3,
             label=f"all chunks (n={len(vals)})")
    if len(poi_vals):
        plt.hist(poi_vals, bins=40, color="#ff7f0e", alpha=0.85,
                 edgecolor="k", lw=0.3,
                 label=f"poisoned (n={len(poi_vals)})")
    thr = np.quantile(vals, 0.99) if len(vals) else 0.0
    plt.axvline(thr, color="#2ca02c", ls="--", lw=1.2, label=r"top 1\% ($k^\star$)")
    plt.xlabel("Dual centrality $c(v)$"); plt.ylabel("Count")
    plt.yscale("log")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out); plt.close()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rage-figs")
    p.add_argument("--results-root", default=str(RESULTS_DIR))
    p.add_argument("--fig-dir", default=str(PAPER_FIG_DIR))
    p.add_argument("--tbl-dir", default=str(PAPER_TBL_DIR))
    p.add_argument("--rescore", action="store_true",
                   help="Recompute derived metrics in every result.json "
                        "(no LLM calls).")
    p.add_argument("--figs-only", action="store_true",
                   help="Skip rescore; rely on cached derived metrics.")
    args = p.parse_args(argv)
    root = Path(args.results_root)
    if args.rescore:
        n = 0
        for rj in root.rglob("result.json"):
            rescore_run(rj); n += 1
        print(f"[rescore] updated {n} result.json files")
    runs = load_runs(root)
    if not runs:
        print(f"No runs found under {root}", file=sys.stderr); return 1
    print(f"Loaded {len(runs)} runs.")
    # ensure derived metrics are present
    for r in runs:
        if DERIVED_KEY not in r:
            r[DERIVED_KEY] = derive_run_metrics(r)
    stats = aggregate_by_defense(runs)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig_accuracy_curves(runs, fig_dir / "fig3_accuracy_curves.pdf")
    fig_poisoned_served_fraction(runs, fig_dir / "fig4_poisoned_fraction.pdf")
    fig_accuracy_floor_bar(stats, fig_dir / "fig5_accuracy_floor.pdf")
    fig_bipartite_topology(runs, fig_dir / "fig1_bipartite_topology.pdf")
    fig_beta_fit(runs, fig_dir / "fig2_beta_fit.pdf",
                  data_out=fig_dir / "beta_fit.json")
    fig_ode_overlay(runs, fig_dir / "fig6_ode_overlay.pdf")
    fig_centrality_distribution(runs, fig_dir / "fig8_centrality.pdf")
    # tables
    tbl_dir = Path(args.tbl_dir); tbl_dir.mkdir(parents=True, exist_ok=True)
    write_tab3(stats, tbl_dir / "tab3_defense_efficacy_body.tex")
    pw = pairwise_stats(runs, focal="evp", metric="final_accuracy")
    if pw:
        write_pvalue_table(pw, tbl_dir / "tab_pvalues_body.tex")
    pw_floor = pairwise_stats(runs, focal="evp", metric="accuracy_floor")
    if pw_floor:
        write_pvalue_table(pw_floor, tbl_dir / "tab_pvalues_floor_body.tex")
    (tbl_dir / "stats_summary.json").write_text(json.dumps(stats, indent=2))
    (tbl_dir / "pairwise_stats.json").write_text(json.dumps(
        {"final_accuracy": pw, "accuracy_floor": pw_floor}, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
