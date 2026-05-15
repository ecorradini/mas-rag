"""`rage-ablations` — leave-one-out + parameter sweep for EVP.

Each ablation is implemented by overriding EVP constructor kwargs:

  - no_ingest_check      : ingest_sim_threshold = 9.0 (never trips)
  - no_centrality        : kstar_frac = 1.0 (verify all retrieved chunks; loses targeting)
  - no_trust_feedback    : trust_threshold = -1.0 (no author-based dropping)
  - no_blacklist         : blacklist_after = 10**9 (effectively disabled)
  - no_trusted_backbone  : trusted_authors = () (corroborators drawn from any author)
  - no_uncorr_quarantine : quarantine_on_uncorroborated = False
  - kstar_sweep          : kstar_frac in {0.05, 0.10, 0.20, 0.40}
  - mn_sweep             : (m,n) in {(1,3),(2,3),(2,5),(3,5)}

Token cost: ~24 runs × ~50k tokens (T=40) ≈ 1.2M tokens.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ..utils.paths import RESULTS_DIR

ABLATIONS = {
    "no_ingest_check":      {"ingest_sim_threshold": 9.0},
    "no_centrality":        {"kstar_frac": 1.0},
    "no_trust_feedback":    {"trust_threshold": -1.0},
    "no_blacklist":         {"blacklist_after": 10**9},
    "no_trusted_backbone":  {"trusted_authors": []},
    "no_uncorr_quarantine": {"quarantine_on_uncorroborated": False},
}

KSTAR_SWEEP = [0.05, 0.10, 0.20, 0.40]
MN_SWEEP = [(1, 3), (2, 3), (2, 5), (3, 5)]


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-ablations")
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--N_A", type=int, default=6)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--results-subdir", default="ablations")
    args = p.parse_args(argv)

    base = ["--T", str(args.T), "--N_A", str(args.N_A),
            "--difficulty", args.difficulty, "--defense", "evp"]
    out_root = RESULTS_DIR / args.results_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, dict]] = list(ABLATIONS.items())
    for f in KSTAR_SWEEP:
        plan.append((f"kstar_{f:.2f}", {"kstar_frac": f}))
    for m, n in MN_SWEEP:
        plan.append((f"mn_{m}_{n}", {"m": m, "n": n}))

    n_done = n_skip = 0
    for tag, override in plan:
        for seed in args.seeds:
            run_id = f"evp_{tag}_s{seed}"
            out_dir = out_root / run_id
            if (out_dir / "result.json").exists():
                n_skip += 1
                continue
            cmd = [sys.executable, "-m", "rag_epidemic.cli.run_cli",
                   "--run-id", run_id, "--seed", str(seed),
                   "--results-dir", str(out_dir),
                   "--evp-override", json.dumps(override), *base]
            if _run(cmd) == 0:
                n_done += 1
    print(f"[ablations] completed={n_done} skipped={n_skip} "
          f"out={out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
