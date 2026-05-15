"""`rage-mixed-encoder` — heterogeneous-encoder probe.

Half of the agents use OpenAI ``text-embedding-3-small``; the other half
use ``text-embedding-3-large``. Implementation hook: the ``RAGE_MIXED_ENCODERS``
environment variable is consumed by ``rag/embeddings.py`` which falls back to
the default encoder when unset.

Token cost: 3 defenses × 3 seeds ≈ 9 runs at T=40 ≈ 0.45M tokens
(plus embedding tokens, billed separately).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from ..utils.paths import RESULTS_DIR


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-mixed-encoder")
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--defenses", nargs="+",
                   default=["undefended", "flp", "evp"])
    p.add_argument("--results-subdir", default="mixed_encoder")
    args = p.parse_args(argv)

    out_root = RESULTS_DIR / args.results_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RAGE_MIXED_ENCODERS"] = "text-embedding-3-small,text-embedding-3-large"

    n_done = n_skip = 0
    for d in args.defenses:
        for seed in args.seeds:
            tag = f"{d}_mix_s{seed}"
            out_dir = out_root / tag
            if (out_dir / "result.json").exists():
                n_skip += 1
                continue
            cmd = [sys.executable, "-m", "rag_epidemic.cli.run_cli",
                   "--run-id", tag, "--defense", d,
                   "--seed", str(seed), "--T", str(args.T),
                   "--results-dir", str(out_dir)]
            print(" ".join(cmd))
            if subprocess.run(cmd, env=env, check=False).returncode == 0:
                n_done += 1
    print(f"[mixed-encoder] completed={n_done} skipped={n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
