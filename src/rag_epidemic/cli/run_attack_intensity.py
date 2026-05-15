"""`rage-attack` — sweep adversary poison rate against the top-3 defenses.

Sweep: poison_rate ∈ {0.25, 0.50, 0.75, 1.00}
Defenses: undefended, FLP, EVP
Seeds: {0,1,2}

Token cost: 4 × 3 × 3 ≈ 36 runs at T=40 ≈ 1.8M tokens.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ..utils.paths import RESULTS_DIR

DEFAULT_DEFENSES = ["undefended", "flp", "evp"]
DEFAULT_INTENSITIES = [0.25, 0.50, 0.75, 1.00]


def _run(cmd: list[str]) -> int:
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-attack")
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--defenses", nargs="+", default=DEFAULT_DEFENSES)
    p.add_argument("--intensities", type=float, nargs="+",
                   default=DEFAULT_INTENSITIES)
    p.add_argument("--N_A", type=int, default=6)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--results-subdir", default="attack_intensity")
    args = p.parse_args(argv)

    out_root = RESULTS_DIR / args.results_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = 0
    for d in args.defenses:
        for rho in args.intensities:
            for seed in args.seeds:
                tag = f"{d}_p{int(rho*100):03d}_s{seed}"
                out_dir = out_root / tag
                if (out_dir / "result.json").exists():
                    n_skip += 1
                    continue
                cmd = [sys.executable, "-m", "rag_epidemic.cli.run_cli",
                       "--run-id", tag, "--defense", d,
                       "--seed", str(seed), "--T", str(args.T),
                       "--N_A", str(args.N_A),
                       "--difficulty", args.difficulty,
                       "--poison-rate", str(rho),
                       "--results-dir", str(out_dir)]
                if _run(cmd) == 0:
                    n_done += 1
    print(f"[attack] completed={n_done} skipped={n_skip} out={out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
