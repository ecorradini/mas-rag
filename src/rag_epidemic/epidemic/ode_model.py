"""Bipartite SI ODE: integrate Ṡ, İ, K̇, Ṗ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ODEParams:
    N_A: int            # number of agents
    N_K: int            # number of chunks at t=0
    beta_star: float    # effective infection rate
    mu: float           # agent recovery rate
    rho_I: float        # mean chunk authorship rate (per infected agent)
    rho_S: float        # mean chunk authorship rate (per healthy agent)
    lambda_p: float     # patient zero injection rate (poisoned chunks/step)


def si_system(t, y, p: ODEParams):
    S, I, K, P = y
    beta = p.beta_star
    # S, I are agent populations; K healthy chunks, P poisoned chunks
    # agent infection: contact between susceptible agent and poisoned chunks (retrieval-mediated)
    K_total = max(K + P, 1.0)
    dS = -beta * S * (P / K_total) + p.mu * I
    dI =  beta * S * (P / K_total) - p.mu * I
    # chunk dynamics: agents write chunks; infected agents write poisoned ones
    dK = p.rho_S * S
    dP = p.rho_I * I + p.lambda_p
    return [dS, dI, dK, dP]


def simulate(p: ODEParams, T: int = 200, I0: int = 1, P0: int = 1) -> dict:
    y0 = [p.N_A - I0, I0, max(p.N_K - P0, 0), P0]
    sol = solve_ivp(si_system, (0, T), y0, args=(p,), dense_output=True,
                    max_step=1.0, method="RK45")
    ts = np.linspace(0, T, T + 1)
    Y = sol.sol(ts)
    return {"t": ts, "S": Y[0], "I": Y[1], "K": Y[2], "P": Y[3]}


def basic_reproduction_number(p: ODEParams, k_A_mean: float, k_K_mean: float) -> float:
    """R0 ≈ √(ρ_I/μ · β* · ⟨k_A⟩⟨k_K⟩)."""
    val = (p.rho_I / max(p.mu, 1e-6)) * p.beta_star * k_A_mean * k_K_mean
    return float(np.sqrt(max(val, 0.0)))
