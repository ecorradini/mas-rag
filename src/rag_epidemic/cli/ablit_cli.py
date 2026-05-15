"""`rage-ablit` — run Heretic-style abliteration."""
from __future__ import annotations

import argparse
import sys

from ..abliteration.heretic_runner import AbliterationConfig, abliterate
from ..abliteration.kl_eval import evaluate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-ablit")
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--top-k-layers", type=int, default=3)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--out-json", default="results/abliteration_kl.json")
    args = p.parse_args(argv)
    cfg = AbliterationConfig(model_id=args.model_id, device=args.device,
                             dtype=args.dtype, top_k_layers=args.top_k_layers)
    out = abliterate(cfg)
    print(f"Saved abliterated model to {out}")
    if args.evaluate:
        rep = evaluate(args.model_id, out, args.out_json,
                       device=args.device, dtype=args.dtype)
        print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
