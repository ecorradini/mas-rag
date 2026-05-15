"""Project-wide paths and environment loading."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

# experiments/src/rag_epidemic/utils/paths.py -> experiments/
EXPERIMENTS_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = EXPERIMENTS_DIR.parent
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CACHE_DIR = EXPERIMENTS_DIR / ".cache"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
PAPER_TBL_DIR = REPO_ROOT / "paper" / "tables"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

for d in (DATA_DIR, RESULTS_DIR, CACHE_DIR, PAPER_FIG_DIR, PAPER_TBL_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """Load .env from experiments/ if present."""
    env_path = EXPERIMENTS_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def get_openai_key() -> str | None:
    load_env()
    return os.environ.get("OPENAI_API_KEY")
