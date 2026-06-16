#!/usr/bin/env python3
"""Cultural-evolution of cooperation in the iterated Donor Game.

Replicates and extends Vallinder & Hughes (2024, arXiv:2412.10270):

  Phase 2  cross-model arm   — Claude 3.5 Sonnet vs GPT-4o vs Gemini 1.5 Flash
                                (resolves the withdrawn H7 cross-model claim)
  Phase 3  grounding arm     — --grounded folds each agent's ESS persona into
                                the donor prompt (Φ/P_LLM dissociation under
                                evolutionary selection)
  Phase 4  cross-cultural    — --culture seeds the initial population from a
                                WVS/ESS cluster trust band

Usage:
    # Dry run — no API, validates the whole pipeline (rule-based decider)
    python scripts/run_evolution.py --dry-run

    # Cross-model MVP (requires ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)
    python scripts/run_evolution.py --models claude,gpt4o,gemini --seeds 5

    # Cost-validation micro-run before the full sweep
    python scripts/run_evolution.py --models claude --generations 2 --agents 4 --seeds 1

    # Grounding × evolution
    python scripts/run_evolution.py --models claude --grounded --seeds 3

Outputs:
    analysis/evolution/<tag>_results.json     (per model/seed trajectories + lineage refs)
    analysis/evolution/<tag>_trajectory.csv   (generation × model cooperation rate)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_OUT_DIR = PROJECT_ROOT / "analysis" / "evolution"

# Model name → ModelConfig classmethod (resolved lazily to avoid heavy imports on --dry-run).
_MODEL_CONFIGS = {
    "claude": "claude_sonnet",
    "gpt4o": "gpt4o",
    "gemini": "gemini_flash",
    "mistral": "mistral_7b",
}


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env into os.environ.

    Tolerant of non-KEY lines (the project's .env contains bare continuation
    tokens that break shell `source`). Does not overwrite already-set vars.
    """
    import os
    import re

    if not path.exists():
        return
    for ln in path.read_text().splitlines():
        m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$", ln)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip("'\"")


def _build_agents(n: int, rng: np.random.Generator, *, culture: str | None, ess_df):
    """Build n donor-game agents. Synthetic personas by default; ESS-sampled per cluster if requested."""
    from agents.agent import Agent
    from agents.memory import HierarchicalMemory
    from agents.profile import AgentProfile
    from agents.state import AgentState
    from decision.mock_policy import MockPolicy

    rows = None
    if culture and ess_df is not None:
        lo, hi = _CLUSTER_BANDS[culture]
        band = ess_df[(ess_df["trust_people"] >= lo) & (ess_df["trust_people"] < hi)]
        if len(band) == 0:
            band = ess_df
        rows = band.sample(n=n, replace=len(band) < n, random_state=int(rng.integers(0, 2**31 - 1)))

    agents = []
    for i in range(n):
        if rows is not None:
            r = rows.iloc[i].to_dict()
            trust = float(r.get("trust_people") or 0.5)
            risk = float(r.get("risk_taking") or 0.5)
            country = r.get("country")
        else:
            trust = float(rng.uniform(0.2, 0.8))
            risk = float(rng.uniform(0.2, 0.8))
            country = None
        profile = AgentProfile(
            agent_id=f"a{i:03d}",
            age=35,
            income=1000.0,
            education="secondary",
            occupation="worker",
            location="urban",
            political_preference="center",
            risk_tolerance=risk,
            social_class="middle",
            trust_people=max(0.0, min(1.0, trust)),
            country=country,
        )
        agents.append(
            Agent(
                profile=profile,
                state=AgentState(wealth=0.0),
                memory=HierarchicalMemory(max_recent=10),
                policy=MockPolicy(),
            )
        )
    return agents


# Cluster trust bands (mirror data/cross_cultural_benchmarks_expanded.json ranges).
_CLUSTER_BANDS = {
    "Eastern": (0.0, 0.44),
    "Southern": (0.44, 0.48),
    "Western": (0.48, 0.53),
    "Anglo": (0.53, 0.60),
    "Northern": (0.60, 0.66),
    "Nordic": (0.66, 1.01),
}


def _make_decider(model: str, *, grounded: bool, dry_run: bool):
    """Return (decision_fn, backend_or_None)."""
    if dry_run or model == "mock":
        from environment.donor_game import reputation_threshold

        return reputation_threshold(0.5), None

    from decision.donor_llm import make_llm_donor_decider
    from decision.model_config import ModelConfig, get_backend

    cfg = getattr(ModelConfig, _MODEL_CONFIGS[model])()
    backend = get_backend(cfg)
    return make_llm_donor_decider(backend, grounded=grounded), backend


def _run_one(model: str, seed: int, args, ess_df) -> dict:
    from simulation.evolution_kernel import EvolutionRunner

    rng = np.random.default_rng(seed)
    agents = _build_agents(args.agents, rng, culture=args.culture, ess_df=ess_df)
    decision_fn, backend = _make_decider(model, grounded=args.grounded, dry_run=args.dry_run)

    runner = EvolutionRunner(
        agents,
        decision_fn=decision_fn,
        n_generations=args.generations,
        rounds_per_generation=args.rounds,
        survivor_fraction=0.5,
        seed=seed,
    )
    result = runner.run()
    usage = backend.usage_report() if backend is not None and hasattr(backend, "usage_report") else None
    return {
        "model": model,
        "seed": seed,
        "cooperation_trajectory": result.cooperation_trajectory,
        "final_cooperation": result.cooperation_trajectory[-1],
        "config": result.config,
        "usage": usage,
        "lineage_size": len(result.lineage),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Cultural evolution of cooperation in the Donor Game.")
    ap.add_argument("--dry-run", action="store_true", help="No API; rule-based decider. Validates pipeline.")
    ap.add_argument("--models", type=str, default="mock", help="Comma list: claude,gpt4o,gemini,mistral,mock.")
    ap.add_argument("--seeds", type=int, default=5, help="Independent runs per model (default 5, as in V&H).")
    ap.add_argument("--generations", type=int, default=10, help="Generations (default 10).")
    ap.add_argument("--agents", type=int, default=12, help="Agents per generation (default 12).")
    ap.add_argument("--rounds", type=int, default=12, help="Donor rounds per generation (default 12).")
    ap.add_argument("--grounded", action="store_true", help="Fold ESS persona into the donor prompt (Phase 3).")
    ap.add_argument(
        "--culture", type=str, default=None, choices=list(_CLUSTER_BANDS), help="Seed from a cluster (Phase 4)."
    )
    ap.add_argument("--tag", type=str, default=None, help="Output filename tag.")
    args = ap.parse_args()

    _load_dotenv(PROJECT_ROOT / ".env")

    if args.dry_run:
        args.models = "mock"
        args.generations = min(args.generations, 4)
        args.agents = min(args.agents, 6)
        args.seeds = min(args.seeds, 2)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = list(range(42, 42 + args.seeds))

    ess_df = None
    if args.culture and not args.dry_run:
        import pandas as pd

        ess_df = pd.read_parquet(PROJECT_ROOT / "data" / "ess_clean.parquet")

    tag = args.tag or (
        ("dryrun" if args.dry_run else "+".join(models))
        + ("_grounded" if args.grounded else "")
        + (f"_{args.culture}" if args.culture else "")
    )

    print("=" * 68)
    print("  Cultural Evolution of Cooperation — Donor Game")
    print(f"  Models: {models}   Seeds: {seeds}")
    print(f"  Gen: {args.generations}  Agents: {args.agents}  Rounds/gen: {args.rounds}")
    print(f"  Grounded: {args.grounded}   Culture: {args.culture}   Mode: {'dry-run' if args.dry_run else 'real'}")
    print("=" * 68)

    runs: list[dict] = []
    failed: dict[str, str] = {}
    for model in models:
        print(f"\n── model={model} ──")
        for seed in seeds:
            try:
                r = _run_one(model, seed, args, ess_df)
            except Exception as exc:  # one model/seed failing must not lose the rest
                failed[f"{model}:s{seed}"] = str(exc)[:200]
                print(f"  seed={seed}  FAILED: {str(exc)[:140]}")
                continue
            runs.append(r)
            traj = ", ".join(f"{x:.2f}" for x in r["cooperation_trajectory"])
            print(f"  seed={seed}  final_coop={r['final_cooperation']:.3f}  traj=[{traj}]")

    # Aggregate per-model mean trajectory ± across seeds (only models with runs).
    summary = {}
    completed_models = [m for m in models if any(r["model"] == m for r in runs)]
    for model in completed_models:
        trajs = np.array([r["cooperation_trajectory"] for r in runs if r["model"] == model])
        summary[model] = {
            "mean_trajectory": trajs.mean(axis=0).round(4).tolist(),
            "std_trajectory": trajs.std(axis=0).round(4).tolist(),
            "mean_final": round(float(trajs[:, -1].mean()), 4),
            "n_seeds": int(trajs.shape[0]),
        }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = _OUT_DIR / f"{tag}_results.json"
    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "seeds": seeds,
            "generations": args.generations,
            "agents": args.agents,
            "rounds_per_generation": args.rounds,
            "grounded": args.grounded,
            "culture": args.culture,
            "dry_run": args.dry_run,
            "reference": "Vallinder & Hughes 2024 (arXiv:2412.10270)",
            "failures": failed,
        },
        "per_model_summary": summary,
        "runs": runs,
    }
    results_path.write_text(json.dumps(payload, indent=2))

    # Trajectory CSV: generation × model mean cooperation (completed models only).
    traj_path = _OUT_DIR / f"{tag}_trajectory.csv"
    header = "generation," + ",".join(completed_models)
    lines = [header]
    for g in range(args.generations):
        row = [str(g)] + [f"{summary[m]['mean_trajectory'][g]:.4f}" for m in completed_models]
        lines.append(",".join(row))
    traj_path.write_text("\n".join(lines) + "\n")

    print(f"\nResults: {results_path}")
    print(f"Trajectory CSV: {traj_path}")
    print("\nPer-model final cooperation (mean over seeds):")
    for m in completed_models:
        print(f"  {m:10s}  {summary[m]['mean_final']:.3f}  (n={summary[m]['n_seeds']})")
    if failed:
        print(f"\n[warning] {len(failed)} run(s) failed: {list(failed)}")


if __name__ == "__main__":
    main()
