"""Lightweight structured logger using rich, with a JSON-Lines option."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

try:
    from rich.logging import RichHandler
    _HANDLER = RichHandler(rich_tracebacks=True, show_path=False)
except ImportError:
    _HANDLER = logging.StreamHandler(sys.stderr)


def get_logger(name: str = "rage", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_HANDLER)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class JsonlWriter:
    """Append-only JSON-Lines event writer."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self._fp.write(json.dumps(event, default=str) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
