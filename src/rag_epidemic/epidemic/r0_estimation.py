"""Empirical R0 and Tc estimation from simulation logs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class BetaFit:
    alpha: float
    tau: float
    n_used: int


def fit_beta_logistic(cos_sims: np.ndarray, infected: np.ndarray) -> BetaFit:
    """Fit β(cos) = σ(α·cos − τ)."""
    if len(cos_sims) < 5 or len(set(infected.tolist())) < 2:
        return BetaFit(alpha=1.0, tau=0.0, n_used=int(len(cos_sims)))
    X = cos_sims.reshape(-1, 1)
    y = infected.astype(int)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    alpha = float(clf.coef_[0, 0])
    tau = float(-clf.intercept_[0])
    return BetaFit(alpha=alpha, tau=tau, n_used=int(len(cos_sims)))


def empirical_R0(infected_series: list[int], window: int = 10) -> float:
    """Simple exponential-growth slope estimate over early window."""
    arr = np.asarray(infected_series, dtype=float)
    if len(arr) < window + 1:
        return float("nan")
    early = arr[: window + 1].clip(min=1e-3)
    log = np.log(early)
    slope = np.polyfit(np.arange(len(log)), log, 1)[0]
    return float(math.exp(slope))


def empirical_Tc(accuracy_series: list[float], threshold: float = 0.4) -> int | None:
    """First superstep at which task accuracy drops below `threshold`."""
    for t, a in enumerate(accuracy_series):
        if a < threshold:
            return t
    return None


def write_estimates(out: Path, **vals) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in vals.items()}, indent=2))
