"""Aggregate per-run results into figures, tables and ad-hoc statistics.

Metric inventory (J8)
---------------------
* ``final_accuracy``                 — last-step accuracy (already in result.json)
* ``collapse_rate``                  — Tc != None across seeds
* ``Tc_median``                      — first step below θ=0.4
* ``post_collapse_accuracy_floor``   — mean accuracy over the last 20% of
                                       steps (or last 10 steps if longer);
                                       this is what separates EVP from
                                       baselines at long horizons.
* ``poisoned_served_fraction``       — per-step n_poisoned_served /
                                       n_retrieved, derived from events.jsonl
                                       (``chunk_retrieved`` events with
                                       ``is_poisoned``). This is the
                                       defense-relevant retrieval metric
                                       (much more discriminative than the
                                       on-disk fraction).
* ``time_to_first_infection``        — first t with infected_agents > 0
* ``agents_ever_infected``           — max(infected_agents) over the run
* ``R0_hat``                         — exponential growth slope over the
                                       first 10 steps of infected_agents
* ``ingest_blocked_count``           — number of ``ingest_blocked`` events

Stat tests (J10)
----------------
* ``wilcoxon_evp_vs(baseline)``      — Wilcoxon signed-rank, paired across
                                       seeds, per (N_A, difficulty) cell.
* ``cliffs_delta``                   — non-parametric effect size.
* ``bootstrap_ci``                   — 95% CI on the mean.

All metrics are functional (no LLM calls) so they can be backfilled against
existing events.jsonl files via ``rage-figs --rescore``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for rj in root.rglob("result.json"):
        try:
            obj = json.loads(rj.read_text())
            obj["_path"] = str(rj)
            obj["_dir"] = str(rj.parent)
            runs.append(obj)
        except Exception:
            pass
    return runs


def iter_events(run_dir: Path) -> Iterable[dict]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        return iter(())
    def _gen():
        with open(p, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    return _gen()


# --------------------------------------------------------------------------- #
# Per-run derived metrics                                                     #
# --------------------------------------------------------------------------- #
def post_collapse_accuracy_floor(steps: list[dict]) -> float:
    if not steps:
        return float("nan")
    n = len(steps)
    tail = max(10, n // 5)
    sub = steps[-tail:]
    return float(np.mean([s["accuracy"] for s in sub]))


def time_to_first_infection(steps: list[dict]) -> int | None:
    for s in steps:
        if int(s.get("infected_agents", 0)) > 0:
            return int(s["t"])
    return None


def agents_ever_infected(steps: list[dict]) -> int:
    if not steps:
        return 0
    return int(max(int(s.get("infected_agents", 0)) for s in steps))


def r0_hat(steps: list[dict], window: int = 10) -> float:
    series = [max(int(s.get("infected_agents", 0)), 0) for s in steps[: window + 1]]
    if len(series) < window + 1:
        return float("nan")
    y = np.asarray(series, dtype=float).clip(min=1e-3)
    log_y = np.log(y)
    slope = float(np.polyfit(np.arange(len(log_y)), log_y, 1)[0])
    return float(math.exp(slope))


def served_poisoned_per_step(run_dir: Path) -> tuple[list[int], list[int]]:
    """Returns (poisoned_served_per_t, total_served_per_t) aligned by t."""
    served: dict[int, int] = {}
    poisoned: dict[int, int] = {}
    for ev in iter_events(run_dir):
        if ev.get("type") != "chunk_retrieved":
            continue
        t = int(ev.get("t", 0))
        served[t] = served.get(t, 0) + 1
        if ev.get("is_poisoned"):
            poisoned[t] = poisoned.get(t, 0) + 1
    if not served:
        return [], []
    T = max(served)
    return ([poisoned.get(t, 0) for t in range(1, T + 1)],
            [served.get(t, 0) for t in range(1, T + 1)])


def ingest_blocked_count(run_dir: Path) -> int:
    n = 0
    for ev in iter_events(run_dir):
        if ev.get("type") == "ingest_blocked":
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Run-level enrichment (J9 rescore)                                           #
# --------------------------------------------------------------------------- #
DERIVED_KEY = "derived"


def derive_run_metrics(run_obj: dict) -> dict:
    steps = run_obj.get("result", {}).get("steps", [])
    run_dir = Path(run_obj.get("_dir") or run_obj.get("config", {}).get("out_dir", "."))
    poisoned, total = served_poisoned_per_step(run_dir)
    served_frac = [p / max(t_, 1) for p, t_ in zip(poisoned, total)] if total else []
    return {
        "post_collapse_accuracy_floor": post_collapse_accuracy_floor(steps),
        "time_to_first_infection": time_to_first_infection(steps),
        "agents_ever_infected": agents_ever_infected(steps),
        "R0_hat": r0_hat(steps),
        "poisoned_served_per_step": poisoned,
        "total_served_per_step": total,
        "poisoned_served_fraction": served_frac,
        "ingest_blocked_count": ingest_blocked_count(run_dir),
    }


def rescore_run(run_path: Path) -> dict:
    """Recompute all derived metrics in-place; no LLM calls."""
    obj = json.loads(run_path.read_text())
    obj["_dir"] = str(run_path.parent)
    obj[DERIVED_KEY] = derive_run_metrics(obj)
    obj.pop("_dir", None)
    run_path.write_text(json.dumps(obj, indent=2, default=str))
    return obj


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _safe_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.std(xs)) if xs else float("nan")


def bootstrap_ci(xs: list[float], n: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(xs, dtype=float)
    boots = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def aggregate_by_defense(runs: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for r in runs:
        d = r["config"]["defense_name"]
        by.setdefault(d, []).append(r)
    out: dict[str, dict] = {}
    for d, rs in by.items():
        accs = [r["result"]["final_accuracy"] for r in rs]
        collapsed = [bool(r["result"]["collapsed"]) for r in rs]
        tcs = [r["result"]["Tc"] for r in rs if r["result"]["Tc"] is not None]
        floor = [r.get(DERIVED_KEY, {}).get("post_collapse_accuracy_floor",
                                            r["result"]["final_accuracy"]) for r in rs]
        ttfi = [r.get(DERIVED_KEY, {}).get("time_to_first_infection") for r in rs]
        aei = [r.get(DERIVED_KEY, {}).get("agents_ever_infected", 0) for r in rs]
        r0h = [r.get(DERIVED_KEY, {}).get("R0_hat", float("nan")) for r in rs]
        cost = sum(r.get("usage", {}).get("prompt_tokens", 0) * 0.15
                   + r.get("usage", {}).get("completion_tokens", 0) * 0.6 for r in rs) / 1e6
        out[d] = {
            "n_runs": len(rs),
            "acc_mean": _safe_mean(accs),
            "acc_std": _safe_std(accs),
            "acc_ci": bootstrap_ci(accs),
            "collapse_rate": float(np.mean(collapsed)) if collapsed else 0.0,
            "Tc_median": float(np.median(tcs)) if tcs else float("nan"),
            "accuracy_floor_mean": _safe_mean(floor),
            "accuracy_floor_std": _safe_std(floor),
            "ttfi_median": float(np.median([x for x in ttfi if x is not None]))
                if any(x is not None for x in ttfi) else float("nan"),
            "agents_ever_infected_mean": _safe_mean(aei),
            "R0_hat_mean": _safe_mean(r0h),
            "cost_usd": float(cost),
        }
    return out


# --------------------------------------------------------------------------- #
# Statistical tests (J10)                                                      #
# --------------------------------------------------------------------------- #
def cliffs_delta(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    """Returns (statistic, p_value); falls back to (nan, 1.0) when degenerate."""
    try:
        from scipy.stats import wilcoxon
        if len(a) != len(b) or len(a) < 2:
            return float("nan"), 1.0
        if all(x == y for x, y in zip(a, b)):
            return float("nan"), 1.0
        stat, p = wilcoxon(a, b, zero_method="zsplit", alternative="greater")
        return float(stat), float(p)
    except Exception:
        return float("nan"), 1.0


def pairwise_stats(runs: list[dict], focal: str = "evp",
                   metric: str = "final_accuracy") -> dict[str, dict]:
    """Pair EVP vs each baseline on the same seed/N_A/difficulty cell.

    Bonferroni correction is applied across non-focal defenses.
    """
    by_def: dict[str, dict[tuple, float]] = {}
    for r in runs:
        cfg = r["config"]
        key = (cfg["seed"], cfg["N_A"], cfg["difficulty"])
        if metric == "final_accuracy":
            val = r["result"]["final_accuracy"]
        elif metric == "accuracy_floor":
            val = r.get(DERIVED_KEY, {}).get("post_collapse_accuracy_floor",
                                             r["result"]["final_accuracy"])
        else:
            val = r.get(DERIVED_KEY, {}).get(metric, float("nan"))
        by_def.setdefault(cfg["defense_name"], {})[key] = float(val)
    if focal not in by_def:
        return {}
    f = by_def[focal]
    baselines = [d for d in by_def if d != focal]
    m = len(baselines) or 1
    out: dict[str, dict] = {}
    for d in baselines:
        b = by_def[d]
        common = sorted(set(f) & set(b))
        a_v = [f[k] for k in common]
        b_v = [b[k] for k in common]
        stat, p = paired_wilcoxon(a_v, b_v)
        delta = cliffs_delta(a_v, b_v)
        out[d] = {
            "n_pairs": len(common),
            "wilcoxon_stat": stat,
            "p_value": p,
            "p_bonferroni": min(1.0, p * m),
            "cliffs_delta": delta,
        }
    return out


# --------------------------------------------------------------------------- #
# LaTeX emitters                                                              #
# --------------------------------------------------------------------------- #
def write_tab3(stats: dict[str, dict], out_path: Path) -> None:
    rows = []
    for d, s in stats.items():
        floor = s.get("accuracy_floor_mean", s["acc_mean"])
        floor_s = s.get("accuracy_floor_std", s["acc_std"])
        r0 = s.get("R0_hat_mean", float("nan"))
        rows.append(
            f"{d.replace('_',' ')} & "
            f"{s['acc_mean']:.2f} $\\pm$ {s['acc_std']:.2f} & "
            f"{floor:.2f} $\\pm$ {floor_s:.2f} & "
            f"{s['collapse_rate']*100:.0f}\\% & "
            f"{'--' if math.isnan(s['Tc_median']) else int(s['Tc_median'])} & "
            f"{'--' if math.isnan(r0) else f'{r0:.2f}'} & "
            f"\\$ {s['cost_usd']:.2f} \\\\"
        )
    out_path.write_text("\n".join(rows))


def write_pvalue_table(pw: dict[str, dict], out_path: Path) -> None:
    rows = []
    for d, s in pw.items():
        rows.append(
            f"{d.replace('_',' ')} & {s['n_pairs']} & "
            f"{s['cliffs_delta']:+.2f} & "
            f"{s['p_value']:.3g} & {s['p_bonferroni']:.3g} \\\\"
        )
    out_path.write_text("\n".join(rows))
