"""`rage-repro` — reproduce the experimental grid with daily token cap+resume.

Key features
------------
* **Idempotent resume**: runs whose ``result.json`` exists are skipped.
* **Per-run persistence**: budget state is updated *after every run* — if you
  Ctrl-C mid-grid you can re-run the exact same command and continue without
  losing the day's accounting.
* **Daily token cap**: stops cleanly once the cap is approached, leaving the
  current run intact. Re-running the command the next day automatically
  resets the daily counter.
* **EMA cost estimation**: predicts the next run's cost from a 4-EMA over
  recently completed runs of the same defense; refuses to start a run whose
  predicted cost would breach the cap.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

from ..defense.factory import ALL_DEFENSES
from ..utils.paths import RESULTS_DIR


def _run(cmd: list[str]) -> int:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _today() -> str:
    return _dt.date.today().isoformat()


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"date": _today(), "tokens_used": 0, "runs": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def _rollover_if_new_day(state: dict) -> dict:
    today = _today()
    if state.get("date") != today:
        return {"date": today, "tokens_used": 0, "runs": []}
    return state


def _read_run_tokens(result_path: Path) -> int:
    try:
        obj = json.loads(result_path.read_text())
        u = obj.get("usage", {})
        return int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
    except Exception:
        return 0


def _ema_estimate(state: dict, defense: str, default: int = 250_000) -> int:
    """4-EMA over the last 4 same-defense runs in the current day's state."""
    same = [r for r in state.get("runs", []) if r.get("defense") == defense]
    if not same:
        all_runs = state.get("runs", [])
        if not all_runs:
            return default
        return int(sum(r["tokens"] for r in all_runs[-4:]) / min(4, len(all_runs)))
    pick = same[-4:]
    return int(sum(r["tokens"] for r in pick) / len(pick))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-repro")
    p.add_argument("--mode", choices=["pilot", "full", "smoke", "ablation",
                                      "attack_intensity", "cross_domain",
                                      "mixed_encoder"], default="pilot")
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--defenses", nargs="+", default=ALL_DEFENSES)
    p.add_argument("--N_A", type=int, nargs="+", default=[6])
    p.add_argument("--difficulties", nargs="+", default=["medium"])
    p.add_argument("--pz-template-only", action="store_true")
    p.add_argument("--daily-token-cap", type=int, default=9_500_000,
                   help="Stop cleanly once predicted day-total > cap. "
                        "Default 9.5M leaves headroom under the 10M/day free tier.")
    p.add_argument("--budget-state-file", default="",
                   help="JSON file persisting the day's token usage. "
                        "Default: experiments/.budget_state.json")
    p.add_argument("--reset-budget", action="store_true",
                   help="Wipe the budget state file before starting.")
    p.add_argument("--results-subdir", default="",
                   help="Override results subdir (default: <mode>)")
    args = p.parse_args(argv)

    if args.mode == "full":
        args.T = 200
        args.seeds = [0, 1, 2, 3, 4]
        args.N_A = [3, 6, 9]
        args.difficulties = ["easy", "medium", "hard"]

    state_path = Path(args.budget_state_file or "experiments/.budget_state.json")
    if args.reset_budget and state_path.exists():
        state_path.unlink()
    state = _load_state(state_path)
    state = _rollover_if_new_day(state)
    _save_state(state_path, state)

    subdir = args.results_subdir or args.mode
    grid = list(product(args.defenses, args.seeds, args.N_A, args.difficulties))
    print(f"[plan] {len(grid)} configurations | mode={args.mode} | T={args.T} | "
          f"day={state['date']} | used={state['tokens_used']:,} | "
          f"cap={args.daily_token_cap:,}")

    n_done, n_skipped, n_capped = 0, 0, 0
    for defense, seed, n_a, diff in grid:
        run_id = f"{defense}_s{seed}_n{n_a}_d{diff}"
        out_dir = RESULTS_DIR / subdir / run_id
        if (out_dir / "result.json").exists():
            n_skipped += 1
            continue

        # rollover check before each run (long grids may span midnight UTC)
        state = _rollover_if_new_day(state)

        est = _ema_estimate(state, defense)
        projected = state["tokens_used"] + est
        if projected > args.daily_token_cap:
            print(f"[cap] would-be projected={projected:,} > cap "
                  f"{args.daily_token_cap:,}. Stopping cleanly; resume tomorrow.")
            n_capped += 1
            break

        print(f"[run] {run_id} | est~{est:,} tok | day-used "
              f"{state['tokens_used']:,}/{args.daily_token_cap:,}")
        cmd = [sys.executable, "-m", "rag_epidemic.cli.run_cli",
               "--run-id", run_id, "--defense", defense,
               "--seed", str(seed), "--T", str(args.T),
               "--N_A", str(n_a), "--difficulty", diff,
               "--results-dir", str(out_dir)]
        if args.pz_template_only:
            cmd.append("--pz-template-only")
        rc = _run(cmd)
        if rc != 0:
            print(f"[fail] {run_id} exited {rc}")
            continue
        # commit actual usage
        actual = _read_run_tokens(out_dir / "result.json")
        state["tokens_used"] += actual
        state["runs"].append({"run_id": run_id, "defense": defense,
                              "tokens": actual, "ts": _dt.datetime.now().isoformat()})
        _save_state(state_path, state)
        n_done += 1
        print(f"[ok]  {run_id} | used {actual:,} | day-total {state['tokens_used']:,}")

    print(f"[done] completed={n_done} skipped={n_skipped} capped={n_capped} "
          f"day-total={state['tokens_used']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
