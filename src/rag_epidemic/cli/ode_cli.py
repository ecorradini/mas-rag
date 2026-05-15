"""`rage-ode` — integrate the SI ODE and print summary stats."""
from __future__ import annotations

import argparse
import json
import sys

from ..epidemic.ode_model import ODEParams, basic_reproduction_number, simulate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-ode")
    p.add_argument("--N_A", type=int, default=6)
    p.add_argument("--N_K", type=int, default=200)
    p.add_argument("--beta-star", type=float, default=0.05)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--rho-I", type=float, default=1.0)
    p.add_argument("--rho-S", type=float, default=1.0)
    p.add_argument("--lambda-p", type=float, default=1.0)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--k-A", type=float, default=5.0)
    p.add_argument("--k-K", type=float, default=2.0)
    args = p.parse_args(argv)
    pp = ODEParams(N_A=args.N_A, N_K=args.N_K, beta_star=args.beta_star,
                   mu=args.mu, rho_I=args.rho_I, rho_S=args.rho_S,
                   lambda_p=args.lambda_p)
    sol = simulate(pp, T=args.T)
    R0 = basic_reproduction_number(pp, args.k_A, args.k_K)
    out = {"R0": R0, "I_final": float(sol["I"][-1]), "P_final": float(sol["P"][-1]),
           "S_final": float(sol["S"][-1])}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
