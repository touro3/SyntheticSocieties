#!/usr/bin/env python3
"""Run cross-cultural ESS validation experiments (Phase 17).

Tests whether BGF agents grounded in higher-trust ESS country clusters
produce higher cooperation rates, recovering the empirical cross-cultural
trust gradient.

Each cluster simulation grounds the population via SocietySpec.trust_people_band
(high / moderate / low), filtering the local AT ESS parquet to approximate the
respective cluster's trust profile.  Published ESS-11 country-level trust means
serve as the empirical benchmarks.

Usage:
    # Mock policy (no GPU, fast)
    python scripts/run_cross_cultural.py

    # Dry run (5 agents, 3 rounds — validates pipeline without ESS parquet)
    python scripts/run_cross_cultural.py --dry-run

    # With LLM (GPU required)
    python scripts/run_cross_cultural.py --include-llm

Outputs:
    analysis/cross_cultural_results.json
    analysis/tables/cross_cultural_correlation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the path — consistent with other scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.cross_cultural import (
    ClusterMultiSeedResult,
    ClusterSimResult,
    CrossCulturalResult,
    compute_cluster_ci,
    compute_cross_cultural_correlation,
    compute_cross_cultural_correlation_multiseed,
    format_cross_cultural_table,
)
from population.country_clusters import CountryCluster, load_clusters

# ── Output paths ──────────────────────────────────────────────────────────────

_RESULTS_PATH = PROJECT_ROOT / "analysis" / "cross_cultural_results.json"
_TABLE_PATH = PROJECT_ROOT / "analysis" / "tables" / "cross_cultural_correlation.csv"
_ESS_PATH = PROJECT_ROOT / "data" / "ess_clean.parquet"
_BENCHMARKS_PATH = PROJECT_ROOT / "data" / "cross_cultural_benchmarks.json"


# ── Agent-builder helper (mirrors run_trust_gradient.py) ──────────────────────


def _make_agent(row: dict, agent_id: str, rng: np.random.Generator, policy):
    """Build a BGF Agent from a grounded ESS profile row."""
    from agents.agent import Agent
    from agents.memory import HierarchicalMemory
    from agents.profile import AgentProfile
    from agents.state import AgentState

    def _sf(val, default=0.5):
        if val is None:
            return default
        try:
            import math

            f = float(val)
            return default if math.isnan(f) else f
        except (TypeError, ValueError):
            return default

    def _si(val, default=35):
        if val is None:
            return default
        try:
            import math

            f = float(val)
            if math.isnan(f):
                return default
            return int(round(f))
        except (TypeError, ValueError):
            return default

    profile = AgentProfile(
        agent_id=agent_id,
        age=_si(row.get("age"), 35),
        income=_sf(row.get("income_decile"), 5.0) * 200,
        education=str(row.get("education_level") or "secondary"),
        occupation="worker",
        location="urban",
        political_preference="center",
        risk_tolerance=_sf(row.get("risk_taking"), 0.5),
        social_class="middle",
        trust_people=max(0.0, min(1.0, _sf(row.get("trust_people"), 0.5))),
        competitiveness=max(0.0, min(1.0, _sf(row.get("competitiveness"), 0.5))),
    )
    state = AgentState(
        wealth=float(rng.uniform(40, 80)),
        stress=float(rng.uniform(0.0, 0.3)),
        satisfaction=_sf(row.get("life_satisfaction"), 0.6),
    )
    return Agent(
        profile=profile,
        state=state,
        memory=HierarchicalMemory(max_recent=10),
        policy=policy,
    )


# ── Simulation runner ─────────────────────────────────────────────────────────


def _run_cluster(
    cluster: CountryCluster,
    n_agents: int,
    n_rounds: int,
    policy_type: str,
    seed: int,
    dry_run: bool,
) -> ClusterSimResult:
    """Run one cluster simulation and return a ClusterSimResult.

    In dry-run mode returns synthetic data so the pipeline can be validated
    without the ESS parquet or GPU.
    """
    t0 = time.time()
    print(f"\n  [{cluster.name}] trust_band={cluster.trust_band}  seed={seed}", end=" ", flush=True)

    if dry_run:
        import random

        rng2 = random.Random(hash(f"{cluster.name}{seed}"))
        # Synthetic: higher-trust cluster → higher cooperation rate
        band_base = {"high": 0.30, "moderate": 0.18, "low": 0.10}
        coop = band_base.get(cluster.trust_band, 0.15) + rng2.uniform(-0.03, 0.03)
        gini = 0.12 + rng2.uniform(0.0, 0.08)
        elapsed = time.time() - t0
        print(f"(dry-run, {elapsed:.1f}s)  coop={coop:.3f}")
        return ClusterSimResult(
            cluster_name=cluster.name,
            ess_mean_trust=cluster.ess_mean_trust,
            simulated_cooperation_rate=round(coop, 4),
            simulated_gini=round(gini, 4),
            n_agents=n_agents,
            n_rounds=n_rounds,
        )

    from bgf_logging.event_logger import EventLogger
    from environment.institutions import InstitutionManager
    from environment.network import NetworkManager
    from environment.world import World
    from environment.world_state import WorldState
    from metrics.event_metrics import behavior_summary_from_events, load_events
    from metrics.inequality import gini_coefficient
    from population.ess_grounding import ESSGrounder
    from population.society_spec import SocietySpec
    from simulation.kernel import SimulationKernel

    rng = np.random.default_rng(seed)

    # Build policy
    if policy_type == "mock":
        from decision.mock_policy import MockPolicy

        policy = MockPolicy()
    elif policy_type == "rule_based":
        from decision.rule_based_policy import RuleBasedPolicy

        policy = RuleBasedPolicy()
    elif policy_type == "llm":
        from decision.llm_backend import LLMBackend
        from decision.llm_policy import LLMPolicy

        backend = LLMBackend.get_instance(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            dtype="float16",
            device_map="auto",
            max_new_tokens=128,
            temperature=0.7,
            inference_timeout=120,
            max_retries=2,
            quantization="4bit",
        )
        backend.load()
        policy = LLMPolicy(backend=backend)
    else:
        raise ValueError(f"Unsupported policy type for cross-cultural run: {policy_type!r}")

    # Ground population via SocietySpec trust band
    spec = SocietySpec(
        narrative=cluster.description,
        trust_people_band=cluster.trust_band,
        target_population_size=n_agents,
    )
    grounder = ESSGrounder(ess_path=_ESS_PATH, min_cohort_size=30)
    ground_result = grounder.ground(spec)
    cohort = ground_result.matched_df

    if len(cohort) < n_agents:
        cohort = cohort.sample(n=n_agents, replace=True, random_state=seed)
    else:
        cohort = cohort.sample(n=n_agents, replace=False, random_state=seed)

    agents = [
        _make_agent(
            row=cohort.iloc[i].to_dict(),
            agent_id=f"{cluster.name}_s{seed}_a{i:04d}",
            rng=rng,
            policy=policy,
        )
        for i in range(n_agents)
    ]

    agent_ids = [a.profile.agent_id for a in agents]
    network = NetworkManager.small_world(agent_ids=agent_ids, k=4, rewiring_prob=0.1, seed=seed)
    world = World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
        network_manager=network,
    )

    # Use a temporary directory for events log
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = Path(tmpdir) / f"{cluster.name}_s{seed}.jsonl"
        logger = EventLogger(str(events_path), overwrite=True)
        kernel = SimulationKernel(agents=agents, world=world, logger=logger)
        kernel.run(num_rounds=n_rounds)

        events = load_events(events_path)
        behavior = behavior_summary_from_events(events)

    coop_rate = float(behavior.get("event_behavior", {}).get("cooperation_rate", 0.0))
    final_wealth = [a.state.wealth for a in agents]
    gini = gini_coefficient(final_wealth) if len(final_wealth) > 1 else 0.0

    elapsed = time.time() - t0
    cohort_size = len(ground_result.matched_df)
    print(f"({elapsed:.0f}s)  coop={coop_rate:.3f}  gini={gini:.3f}  cohort={cohort_size}")

    return ClusterSimResult(
        cluster_name=cluster.name,
        ess_mean_trust=cluster.ess_mean_trust,
        simulated_cooperation_rate=round(coop_rate, 4),
        simulated_gini=round(gini, 4),
        n_agents=n_agents,
        n_rounds=n_rounds,
    )


# ── Results I/O ───────────────────────────────────────────────────────────────


def _save_results(path: Path, results: list[ClusterSimResult], cc_result: CrossCulturalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_results": [
            {
                "cluster_name": r.cluster_name,
                "ess_mean_trust": r.ess_mean_trust,
                "simulated_cooperation_rate": r.simulated_cooperation_rate,
                "simulated_gini": r.simulated_gini,
                "n_agents": r.n_agents,
                "n_rounds": r.n_rounds,
            }
            for r in results
        ],
        "correlation": {
            "pearson_r": round(cc_result.pearson_r, 4),
            "pearson_p": round(cc_result.pearson_p, 4),
            "spearman_rho": round(cc_result.spearman_rho, 4),
            "spearman_p": round(cc_result.spearman_p, 4),
            "gradient_recovered": cc_result.gradient_recovered,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def _save_csv(path: Path, results: list[ClusterSimResult], cc_result: CrossCulturalResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "cluster_name,ess_mean_trust,simulated_cooperation_rate,simulated_gini,"
        "n_agents,n_rounds,pearson_r,spearman_rho,gradient_recovered"
    ]
    for r in sorted(results, key=lambda x: x.ess_mean_trust):
        lines.append(
            f"{r.cluster_name},{r.ess_mean_trust},{r.simulated_cooperation_rate},"
            f"{r.simulated_gini},{r.n_agents},{r.n_rounds},"
            f"{round(cc_result.pearson_r, 4)},{round(cc_result.spearman_rho, 4)},"
            f"{cc_result.gradient_recovered}"
        )
    path.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 — Cross-Cultural ESS Validation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data — validates pipeline without ESS parquet or GPU.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Use LLM policy instead of mock (requires GPU with Mistral-7B).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=20,
        help="Agents per cluster simulation (default: 20).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Simulation rounds per cluster (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42). Seeds used are seed, seed+1, … seed+n_seeds-1.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds to run per cluster (default: 1). Use ≥3 for CIs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_RESULTS_PATH,
        help="Output JSON path for results.",
    )
    args = parser.parse_args()

    # Dry-run overrides: smaller params
    n_agents = 5 if args.dry_run else args.agents
    n_rounds = 3 if args.dry_run else args.rounds
    policy_type = "llm" if args.include_llm else "mock"
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    print("=" * 60)
    print("  Phase 17 — Cross-Cultural ESS Validation")
    print(f"  Mode:    {'dry-run' if args.dry_run else 'real'}")
    print(f"  Policy:  {policy_type}")
    print(f"  Agents:  {n_agents}   Rounds: {n_rounds}   Seeds: {seeds}")
    print("=" * 60)

    clusters = load_clusters(_BENCHMARKS_PATH)

    if args.n_seeds == 1:
        # Single-seed path (backward-compatible)
        cluster_results: list[ClusterSimResult] = []
        for cluster in clusters:
            result = _run_cluster(
                cluster=cluster,
                n_agents=n_agents,
                n_rounds=n_rounds,
                policy_type=policy_type,
                seed=seeds[0],
                dry_run=args.dry_run,
            )
            cluster_results.append(result)

        cc_result = compute_cross_cultural_correlation(cluster_results)

        _save_results(args.out, cluster_results, cc_result)
        print(f"\nResults saved to: {args.out}")
        _save_csv(_TABLE_PATH, cluster_results, cc_result)
        print(f"Table saved to:   {_TABLE_PATH}")
        print()
        print(format_cross_cultural_table(cc_result))

    else:
        # Multi-seed path — aggregate cooperation rates per cluster then correlate
        from collections import defaultdict

        per_cluster_runs: dict[str, list[ClusterSimResult]] = defaultdict(list)

        for seed_idx, seed in enumerate(seeds):
            if seed_idx > 0 and policy_type == "llm":
                from decision.llm_backend import LLMBackend

                LLMBackend.between_seeds()
            print(f"\n── Seed {seed} ──")
            for cluster in clusters:
                result = _run_cluster(
                    cluster=cluster,
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    policy_type=policy_type,
                    seed=seed,
                    dry_run=args.dry_run,
                )
                per_cluster_runs[cluster.name].append(result)

        # Build ClusterMultiSeedResult list for CI-aware correlation
        # compute_cluster_ci() takes the list of ClusterSimResult objects (not floats)
        multi_seed_results: list[ClusterMultiSeedResult] = []
        for cluster in clusters:
            runs = per_cluster_runs[cluster.name]
            multi_seed_results.append(compute_cluster_ci(runs))

        cc_result = compute_cross_cultural_correlation_multiseed(multi_seed_results)

        # Use mean cooperation rates for single-result-compatible outputs
        flat_results = [
            ClusterSimResult(
                cluster_name=r.cluster_name,
                ess_mean_trust=r.ess_mean_trust,
                simulated_cooperation_rate=round(r.mean_cooperation_rate, 4),
                simulated_gini=round(r.mean_gini, 4),
                n_agents=r.n_agents,
                n_rounds=r.n_rounds,
            )
            for r in multi_seed_results
        ]

        _save_results(args.out, flat_results, cc_result)
        print(f"\nResults saved to: {args.out}")
        _save_csv(_TABLE_PATH, flat_results, cc_result)
        print(f"Table saved to:   {_TABLE_PATH}")
        print()
        print(format_cross_cultural_table(cc_result))
        print(
            f"\n[multi-seed] Pearson r = {cc_result.pearson_r:.4f}  "
            f"Spearman ρ = {cc_result.spearman_rho:.4f}  "
            f"({args.n_seeds} seeds per cluster)"
        )


if __name__ == "__main__":
    main()
