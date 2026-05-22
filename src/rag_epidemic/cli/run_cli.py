"""`rage-run` — run a single experiment configuration."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..agents.healthy_agent import HealthyAgent
from ..agents.patient_zero import PatientZero, PatientZeroConfig, default_abliterated_path
from ..defense.factory import build_defense
from ..rag.chroma_store import ChromaStore
from ..rag.embeddings import OpenAIEmbedder
from ..simulation.orchestrator import Orchestrator, RunConfig
from ..tasks.realm_bench import make_task
from ..utils.openai_client import OpenAIClient, TokenBudgetExceeded, billable_usage, usage
from ..utils.paths import RESULTS_DIR, load_env
from ..utils.seeding import seed_everything as set_global_seed
from ..utils.logging import get_logger

log = get_logger("run")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rage-run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--defense", default="undefended")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--N_A", type=int, default=6)
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--n-warehouses", type=int, default=4)
    p.add_argument("--k-retrieve", type=int, default=5)
    p.add_argument("--no-patient-zero", action="store_true")
    p.add_argument("--pz-template-only", action="store_true",
                   help="Skip loading the local LLM; use template-only adversary")
    p.add_argument("--abliterated-path", default="")
    p.add_argument("--results-dir", default="")
    p.add_argument("--poison-rate", type=float, default=1.0,
                   help="Patient Zero poisoning probability per superstep.")
    p.add_argument("--evp-override", default="",
                   help="JSON dict of EVP constructor overrides "
                        "(e.g. '{\"kstar_frac\":1.0,\"trusted_authors\":[]}').")
    p.add_argument("--run-token-cap", type=int, default=0,
                   help="Maximum billable OpenAI tokens for this single run. "
                        "0 disables the per-run hard cap.")
    args = p.parse_args(argv)

    load_env()
    set_global_seed(args.seed)
    out_dir = Path(args.results_dir or (RESULTS_DIR / args.run_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        run_id=args.run_id, T=args.T, N_A=args.N_A,
        k_retrieve=args.k_retrieve, defense_name=args.defense,
        seed=args.seed, difficulty=args.difficulty,
        n_warehouses=args.n_warehouses, out_dir=str(out_dir),
        enable_patient_zero=not args.no_patient_zero,
        pz_use_llm=not args.pz_template_only,
        poison_rate=args.poison_rate,
    )

    client = OpenAIClient(token_cap=args.run_token_cap)
    embedder = OpenAIEmbedder(client=client)
    store = ChromaStore(path=out_dir / "chroma", name=args.run_id, reset=True)

    # build a balanced agent roster across roles
    roles = ["demand", "inventory", "routing"]
    agents = [HealthyAgent(agent_id=f"agent_{i:02d}",
                           role=roles[i % len(roles)],
                           client=client, seed=args.seed + i)
              for i in range(args.N_A)]

    pz: PatientZero | None = None
    if cfg.enable_patient_zero:
        pz_path = args.abliterated_path or str(default_abliterated_path())
        pz_cfg = PatientZeroConfig(
            model_path=pz_path if Path(pz_path).exists() else "",
            poison_rate=1.0,
            fallback_to_template=True,
        )
        if args.pz_template_only:
            pz_cfg.fallback_to_template = True
            # force no model load
            pz_cfg.model_path = "__none__"
        pz = PatientZero(pz_cfg, agent_id="agent_pz", seed=args.seed)

    defense_kwargs: dict = {}
    if args.evp_override and args.defense == "evp":
        defense_kwargs = json.loads(args.evp_override)
        if "trusted_authors" in defense_kwargs:
            defense_kwargs["trusted_authors"] = tuple(defense_kwargs["trusted_authors"])
    defense = build_defense(args.defense, client=client, **defense_kwargs)
    task = make_task(seed=args.seed, difficulty=args.difficulty,
                     n_warehouses=args.n_warehouses)
    orch = Orchestrator(cfg=cfg, store=store, embedder=embedder, agents=agents,
                        defense=defense, patient_zero=pz, task=task)
    capped_error = ""
    try:
        result = orch.run()
    except TokenBudgetExceeded as exc:
        capped_error = str(exc)
        log.error("Run token cap reached: %s", capped_error)
        result = orch.result
        try:
            orch._fp.close()
        except Exception:
            pass
    u = usage()
    bu = billable_usage()
    result.usage_cost_usd = u.cost_usd()
    payload = {"config": asdict(cfg), "result": {
            **{k: v for k, v in result.__dict__.items() if k != "steps"},
            "steps": [s.__dict__ for s in result.steps],
        }, "usage": u.__dict__, "billable_usage": bu.__dict__}
    if capped_error:
        payload["capped"] = True
        payload["cap_error"] = capped_error
        (out_dir / "capped.json").write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps({"run_id": args.run_id, "capped": True,
                          "cap_error": capped_error,
                          "billable_tokens": bu.prompt_tokens + bu.completion_tokens
                          + bu.embedding_tokens}, indent=2))
        return 2
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"run_id": args.run_id, "final_accuracy": result.final_accuracy,
                      "collapsed": result.collapsed, "Tc": result.Tc,
                      "cost_usd": result.usage_cost_usd,
                      "billable_tokens": bu.prompt_tokens + bu.completion_tokens
                      + bu.embedding_tokens}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
