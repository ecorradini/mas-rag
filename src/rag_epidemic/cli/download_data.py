"""`rage-data` — verify environment and prefetch local model weights.

Datasets (REALM-Bench / OrgForge prose) are *generated* deterministically from
the seeded SimEngine, so there is nothing external to download for the corpus.
What we DO need:
  - Verify OPENAI_API_KEY is set.
  - Optionally pre-download the Llama-3.2-3B base model weights from HF
    (gated; requires HF_TOKEN).
"""

from __future__ import annotations

import argparse
import os
import sys

from ..utils.paths import DATA_DIR, get_openai_key, load_env
from ..utils.logging import get_logger

log = get_logger("data")


def _check_openai() -> bool:
    load_env()
    if not get_openai_key():
        print("[FAIL] OPENAI_API_KEY missing. Add it to experiments/.env.", file=sys.stderr)
        return False
    print("[OK] OPENAI_API_KEY found.")
    return True


def _prefetch_llama(model_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[SKIP] huggingface_hub not available: {e}")
        return False
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        print("[WARN] No HF_TOKEN set; Llama-3.2 is a gated model. "
              "Either accept the licence + add HF_TOKEN to .env, "
              "or run abliteration on a non-gated alternative.")
    target = DATA_DIR / "models" / model_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=model_id, local_dir=str(target),
                          local_dir_use_symlinks=False,
                          token=tok)
        print(f"[OK] Prefetched {model_id} → {target}")
        return True
    except Exception as e:
        print(f"[FAIL] Could not download {model_id}: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-data")
    p.add_argument("--all", action="store_true", help="run every step")
    p.add_argument("--prefetch-llama", action="store_true")
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    args = p.parse_args(argv)
    if not (args.all or args.prefetch_llama):
        args.all = True
    ok = _check_openai()
    if args.all or args.prefetch_llama:
        _prefetch_llama(args.model_id)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
