#!/usr/bin/env python3
"""Expanded cross-cultural ESS validation — 6 clusters, multi-seed, robustness sweep.

Extends Phase 27 (run_cross_cultural.py) with:
- 6 cultural clusters (Eastern / Southern / Western / Anglo / Northern / Nordic)
- N=20 seeds per cluster → mean cooperation ± 95% CI
- Optional robustness sweep across multiple agent counts (20 / 100 / 500)
- Explicit WVS holdout evaluation (out-of-sample benchmark)
- Optional ungrounded control run for grounded-vs-control holdout deltas

Published ESS-11 country-level trust means serve as empirical X-axis benchmarks.
Trust ranges in the local AT parquet proxy each cluster's trust profile.

Usage:
    # Rule-based policy, 20 seeds, 3 agent sizes (fast, no GPU)
    python scripts/run_cross_cultural_expanded.py

    # Dry run (3 seeds, 5 agents — validates pipeline)
    python scripts/run_cross_cultural_expanded.py --dry-run

    # LLM policy, 20 seeds (GPU required)
    python scripts/run_cross_cultural_expanded.py --include-llm

    # Custom seeds and agent count (no robustness sweep)
    python scripts/run_cross_cultural_expanded.py --n-seeds 30 --agents 50 --no-robustness

    # Grounded vs ungrounded control comparison on WVS holdout
    python scripts/run_cross_cultural_expanded.py --run-ungrounded-control

Outputs:
    analysis/cross_cultural_expanded_results.json
    analysis/tables/cross_cultural_expanded_correlation.csv
    analysis/figures/cross_cultural_expanded.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.cross_cultural import (
    ClusterMultiSeedResult,
    ClusterSimResult,
    compare_holdout_fit,
    compute_benchmark_fit,
    compute_cluster_ci,
    compute_cross_cultural_correlation_multiseed,
    format_cross_cultural_table,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_BENCHMARKS_PATH = PROJECT_ROOT / "data" / "cross_cultural_benchmarks_expanded.json"
_ESS_PATH = PROJECT_ROOT / "data" / "ess_clean.parquet"
_RESULTS_PATH = PROJECT_ROOT / "analysis" / "cross_cultural_expanded_results.json"
_TABLE_PATH = PROJECT_ROOT / "analysis" / "tables" / "cross_cultural_expanded_correlation.csv"
_FIGURE_PATH = PROJECT_ROOT / "analysis" / "figures" / "cross_cultural_expanded.png"


# ── Cluster definition (loaded from JSON) ─────────────────────────────────────


@dataclass
class ExpandedCluster:
    name: str
    countries: list[str]
    ess_mean_trust: float
    ess_sd_trust: float
    trust_lo: float
    trust_hi: float
    wvs_trust_pct: float
    description: str


def load_expanded_clusters(path: Path = _BENCHMARKS_PATH) -> list[ExpandedCluster]:
    data = json.loads(path.read_text())
    clusters = []
    for name, c in data["clusters"].items():
        clusters.append(
            ExpandedCluster(
                name=name,
                countries=c["countries"],
                ess_mean_trust=c["ess_mean_trust_people"],
                ess_sd_trust=c["ess_sd_trust_people"],
                trust_lo=c["trust_lo"],
                trust_hi=c["trust_hi"],
                wvs_trust_pct=c["wvs_trust_pct"],
                description=c["description"],
            )
        )
    # Sort ascending by ESS mean (for canonical Spearman ordering)
    return sorted(clusters, key=lambda c: c.ess_mean_trust)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_seeds_arg(raw: str | None, n_seeds_fallback: int) -> list[int]:
    if raw:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not vals:
            raise ValueError("--seeds was provided but no valid integer seeds were parsed.")
        return vals
    return list(range(42, 42 + n_seeds_fallback))


def _assert_no_wvs_columns(ess_df) -> None:
    """Guard against accidental holdout leakage into grounding inputs."""
    cols = {str(c).strip().lower() for c in ess_df.columns}
    forbidden = {"wvs", "wvs_trust", "wvs_trust_pct", "world_values_survey"}
    leaked = sorted(cols & forbidden)
    if leaked:
        raise ValueError(
            f"Potential holdout leakage detected: ESS grounding dataframe contains WVS-like columns: {leaked}"
        )


# ── Agent builder ─────────────────────────────────────────────────────────────


def _make_agent(row: dict, agent_id: str, rng: np.random.Generator, policy):
    import math

    from agents.agent import Agent
    from agents.memory import HierarchicalMemory
    from agents.profile import AgentProfile
    from agents.state import AgentState

    def _sf(val, default=0.5):
        if val is None:
            return default
        try:
            f = float(val)
            return default if math.isnan(f) else f
        except (TypeError, ValueError):
            return default

    def _si(val, default=35):
        if val is None:
            return default
        try:
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


# ── Single simulation run ─────────────────────────────────────────────────────


def _run_single(
    cluster: ExpandedCluster,
    n_agents: int,
    n_rounds: int,
    policy_type: str,
    seed: int,
    ess_df,
    dry_run: bool,
    grounded: bool = True,
) -> ClusterSimResult:
    """Run one cluster simulation for one seed. Returns ClusterSimResult."""
    if dry_run:
        import random

        rng2 = random.Random(hash(f"{cluster.name}{seed}{n_agents}"))
        # Synthetic dry-run behavior:
        # grounded -> trust-conditioned gradient
        # ungrounded control -> near-flat baseline across clusters
        if grounded:
            base = 0.10 + (cluster.ess_mean_trust - 0.40) * 0.55
        else:
            base = 0.16
        coop = max(0.02, min(0.95, base + rng2.gauss(0, 0.02)))
        gini = 0.12 + rng2.uniform(0.0, 0.08)
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

    rng = np.random.default_rng(seed)

    # Build policy (policy objects are stateless for rule_based/mock, shareable)
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
        raise ValueError(f"Unsupported policy: {policy_type!r}")

    # Grounded: filter AT sample to the cluster trust band.
    # Control: sample from the full AT cohort (no cluster conditioning).
    _assert_no_wvs_columns(ess_df)
    if grounded:
        cohort_df = ess_df[(ess_df["trust_people"] >= cluster.trust_lo) & (ess_df["trust_people"] < cluster.trust_hi)]
    else:
        cohort_df = ess_df

    if len(cohort_df) == 0:
        # Fallback: use rows closest to the trust-band midpoint.
        target = (cluster.trust_lo + cluster.trust_hi) / 2.0
        cohort_df = (
            ess_df.assign(_trust_dist=(ess_df["trust_people"] - target).abs())
            .nsmallest(max(10, n_agents), "_trust_dist")
            .drop(columns=["_trust_dist"])
        )

    if len(cohort_df) < n_agents:
        sample = cohort_df.sample(n=n_agents, replace=True, random_state=seed)
    else:
        sample = cohort_df.sample(n=n_agents, replace=False, random_state=seed)

    agents = [
        _make_agent(
            row=sample.iloc[i].to_dict(),
            agent_id=f"{cluster.name}_n{n_agents}_s{seed}_a{i:04d}",
            rng=rng,
            policy=policy,
        )
        for i in range(n_agents)
    ]

    agent_ids = [a.profile.agent_id for a in agents]
    k = min(4, n_agents - 1)  # small-world k must be < n_agents
    network = NetworkManager.small_world(agent_ids=agent_ids, k=k, rewiring_prob=0.1, seed=seed)
    world = World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
        network_manager=network,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = Path(tmpdir) / f"{cluster.name}_n{n_agents}_s{seed}.jsonl"
        logger = EventLogger(str(events_path), overwrite=True)
        from simulation.kernel import SimulationKernel

        kernel = SimulationKernel(agents=agents, world=world, logger=logger)
        kernel.run(num_rounds=n_rounds)
        events = load_events(events_path)
        behavior = behavior_summary_from_events(events)

    coop = float(behavior.get("event_behavior", {}).get("cooperation_rate", 0.0))
    gini = gini_coefficient([a.state.wealth for a in agents]) if len(agents) > 1 else 0.0

    return ClusterSimResult(
        cluster_name=cluster.name,
        ess_mean_trust=cluster.ess_mean_trust,
        simulated_cooperation_rate=round(coop, 4),
        simulated_gini=round(gini, 4),
        n_agents=n_agents,
        n_rounds=n_rounds,
    )


# ── Multi-seed runner for one cluster × one agent size ────────────────────────


def run_cluster_multiseed(
    cluster: ExpandedCluster,
    n_agents: int,
    n_rounds: int,
    policy_type: str,
    seeds: list[int],
    ess_df,
    dry_run: bool,
    grounded: bool = True,
    verbose: bool = True,
) -> ClusterMultiSeedResult:
    single_results: list[ClusterSimResult] = []
    for seed_idx, seed in enumerate(seeds):
        if seed_idx > 0 and policy_type == "llm":
            from decision.llm_backend import LLMBackend

            LLMBackend.between_seeds()
        if verbose:
            print(f"    seed={seed}", end=" ", flush=True)
        r = _run_single(
            cluster=cluster,
            n_agents=n_agents,
            n_rounds=n_rounds,
            policy_type=policy_type,
            seed=seed,
            ess_df=ess_df,
            dry_run=dry_run,
            grounded=grounded,
        )
        single_results.append(r)
        if verbose:
            print(f"coop={r.simulated_cooperation_rate:.3f}", end="  ")
    if verbose:
        print()
    result = compute_cluster_ci(single_results)
    result.wvs_trust_pct = cluster.wvs_trust_pct
    return result


# ── Save / load ───────────────────────────────────────────────────────────────


def _save_expanded(
    path: Path,
    multi_results_by_size: dict[int, list[ClusterMultiSeedResult]],
    n_rounds: int,
    n_seeds: int,
    control_results_by_size: dict[int, list[ClusterMultiSeedResult]] | None = None,
    *,
    seeds: list[int] | None = None,
    policy_type: str = "rule_based",
    dry_run: bool = False,
    benchmarks_path: Path = _BENCHMARKS_PATH,
    ess_path: Path = _ESS_PATH,
    ess_row_count: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _fit_to_dict(fit) -> dict:
        return {
            "benchmark": fit.benchmark_name,
            "pearson_r": round(fit.pearson_r, 4),
            "pearson_p": round(fit.pearson_p, 4),
            "spearman_rho": round(fit.spearman_rho, 4),
            "spearman_p": round(fit.spearman_p, 4),
            "gradient_recovered": fit.gradient_recovered,
            "mae": round(fit.mae, 4),
            "rmse": round(fit.rmse, 4),
            "n_clusters": fit.n_clusters,
        }

    agent_sizes = sorted(multi_results_by_size.keys())
    primary_size = agent_sizes[0]
    primary = multi_results_by_size[primary_size]

    cc_ess = compute_cross_cultural_correlation_multiseed(primary)
    fit_ess = compute_benchmark_fit(primary, benchmark="ess")
    fit_wvs = compute_benchmark_fit(primary, benchmark="wvs")

    payload: dict = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_clusters": len(primary),
            "n_seeds": n_seeds,
            "seeds": seeds or [],
            "n_rounds": n_rounds,
            "agent_sizes_tested": agent_sizes,
            "primary_agent_size": primary_size,
            "contains_ungrounded_control": control_results_by_size is not None,
            "policy_type": policy_type,
            "dry_run": dry_run,
            "data_provenance": {
                "benchmarks_path": str(benchmarks_path),
                "benchmarks_sha256": _sha256_file(benchmarks_path),
                "ess_path": str(ess_path),
                "ess_sha256": None if dry_run else _sha256_file(ess_path),
                "ess_row_count": None if dry_run else ess_row_count,
            },
            "leakage_controls": {
                "grounding_data_source": "ESS only (data/ess_clean.parquet)",
                "evaluation_benchmarks": ["ESS (in-sample)", "WVS (holdout)"],
                "wvs_used_for_agent_grounding": False,
                "wvs_used_only_for_evaluation": True,
            },
        },
        # Backward-compatible alias: historically this field referred to ESS benchmark.
        "correlation": {
            "pearson_r": round(cc_ess.pearson_r, 4),
            "pearson_p": round(cc_ess.pearson_p, 4),
            "spearman_rho": round(cc_ess.spearman_rho, 4),
            "spearman_p": round(cc_ess.spearman_p, 4),
            "gradient_recovered": cc_ess.gradient_recovered,
        },
        "benchmark_fit": {
            "ess": _fit_to_dict(fit_ess),
            "wvs": _fit_to_dict(fit_wvs),
        },
        "cluster_results": {},
        "robustness": {},
    }

    # Primary results
    for r in primary:
        payload["cluster_results"][r.cluster_name] = {
            "ess_mean_trust": r.ess_mean_trust,
            "wvs_trust_pct": r.wvs_trust_pct,
            "mean_cooperation_rate": r.mean_cooperation_rate,
            "std_cooperation_rate": r.std_cooperation_rate,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "n_seeds": r.n_seeds,
            "mean_gini": r.mean_gini,
            "n_agents": r.n_agents,
            "n_rounds": r.n_rounds,
            "seed_cooperation_rates": r.seed_cooperation_rates,
        }

    # Robustness by agent size
    for sz, results in multi_results_by_size.items():
        cc_sz = compute_cross_cultural_correlation_multiseed(results)
        fit_ess_sz = compute_benchmark_fit(results, benchmark="ess")
        fit_wvs_sz = compute_benchmark_fit(results, benchmark="wvs")
        payload["robustness"][str(sz)] = {
            "pearson_r": round(cc_sz.pearson_r, 4),
            "spearman_rho": round(cc_sz.spearman_rho, 4),
            "gradient_recovered": cc_sz.gradient_recovered,
            "ess_fit": _fit_to_dict(fit_ess_sz),
            "wvs_fit": _fit_to_dict(fit_wvs_sz),
            "cluster_mean_coops": {r.cluster_name: r.mean_cooperation_rate for r in results},
        }

    if control_results_by_size is not None:
        control_primary = control_results_by_size[primary_size]
        control_ess = compute_benchmark_fit(control_primary, benchmark="ess")
        control_wvs = compute_benchmark_fit(control_primary, benchmark="wvs")
        wvs_cmp = compare_holdout_fit(primary, control_primary, benchmark="wvs")
        ess_cmp = compare_holdout_fit(primary, control_primary, benchmark="ess")

        payload["ungrounded_control"] = {
            "benchmark_fit": {
                "ess": _fit_to_dict(control_ess),
                "wvs": _fit_to_dict(control_wvs),
            },
            "cluster_results": {
                r.cluster_name: {
                    "ess_mean_trust": r.ess_mean_trust,
                    "wvs_trust_pct": r.wvs_trust_pct,
                    "mean_cooperation_rate": r.mean_cooperation_rate,
                    "std_cooperation_rate": r.std_cooperation_rate,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "n_seeds": r.n_seeds,
                    "mean_gini": r.mean_gini,
                    "n_agents": r.n_agents,
                    "n_rounds": r.n_rounds,
                    "seed_cooperation_rates": r.seed_cooperation_rates,
                }
                for r in control_primary
            },
            "robustness": {},
        }

        for sz, results in control_results_by_size.items():
            payload["ungrounded_control"]["robustness"][str(sz)] = {
                "ess_fit": _fit_to_dict(compute_benchmark_fit(results, benchmark="ess")),
                "wvs_fit": _fit_to_dict(compute_benchmark_fit(results, benchmark="wvs")),
                "cluster_mean_coops": {r.cluster_name: r.mean_cooperation_rate for r in results},
            }

        payload["condition_comparison"] = {
            "wvs": {
                "delta_pearson_r": wvs_cmp.delta_pearson_r,
                "delta_spearman_rho": wvs_cmp.delta_spearman_rho,
                "delta_mae": wvs_cmp.delta_mae,
                "delta_rmse": wvs_cmp.delta_rmse,
                "grounded_better": wvs_cmp.grounded_better,
            },
            "ess": {
                "delta_pearson_r": ess_cmp.delta_pearson_r,
                "delta_spearman_rho": ess_cmp.delta_spearman_rho,
                "delta_mae": ess_cmp.delta_mae,
                "delta_rmse": ess_cmp.delta_rmse,
                "grounded_better": ess_cmp.grounded_better,
            },
        }

    path.write_text(json.dumps(payload, indent=2))


def _save_csv(
    path: Path,
    multi_results: list[ClusterMultiSeedResult],
    cc_ess,
    wvs_fit,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "cluster,ess_mean_trust,wvs_trust_pct,mean_coop,ci_lower,ci_upper,"
        "std_coop,n_seeds,mean_gini,n_agents,pearson_r_ess,spearman_rho_ess,"
        "pearson_r_wvs,spearman_rho_wvs,gradient_ess,gradient_wvs"
    ]
    for r in sorted(multi_results, key=lambda x: x.ess_mean_trust):
        lines.append(
            f"{r.cluster_name},{r.ess_mean_trust},{r.wvs_trust_pct or ''},"
            f"{r.mean_cooperation_rate},{r.ci_lower},{r.ci_upper},"
            f"{r.std_cooperation_rate},{r.n_seeds},{r.mean_gini},{r.n_agents},"
            f"{round(cc_ess.pearson_r, 4)},{round(cc_ess.spearman_rho, 4)},"
            f"{round(wvs_fit.pearson_r, 4)},{round(wvs_fit.spearman_rho, 4)},"
            f"{cc_ess.gradient_recovered},{wvs_fit.gradient_recovered}"
        )
    path.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expanded cross-cultural ESS validation: 6 clusters, multi-seed, robustness sweep"
    )
    parser.add_argument("--dry-run", action="store_true", help="Synthetic data — no ESS parquet or GPU needed.")
    parser.add_argument("--include-llm", action="store_true", help="LLM policy (Mistral-7B, GPU required).")
    parser.add_argument("--n-seeds", type=int, default=20, help="Seeds per cluster (default: 20).")
    parser.add_argument("--agents", type=int, default=20, help="Primary agent count (default: 20).")
    parser.add_argument("--rounds", type=int, default=10, help="Simulation rounds per run (default: 10).")
    parser.add_argument("--no-robustness", action="store_true", help="Skip multi-agent-size robustness sweep.")
    parser.add_argument(
        "--agent-sizes", type=str, default="20,100,500", help="Agent sizes for robustness sweep (default: 20,100,500)."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional explicit seed list (comma-separated), e.g. 42,123,7.",
    )
    parser.add_argument(
        "--run-ungrounded-control",
        action="store_true",
        help="Also run an ungrounded control (no cluster trust filtering) and report deltas.",
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    policy_type = "llm" if args.include_llm else "rule_based"
    n_seeds = 3 if dry_run else args.n_seeds
    n_agents_primary = 5 if dry_run else args.agents
    n_rounds = 3 if dry_run else args.rounds

    agent_sizes_raw = [5] if dry_run else [int(x) for x in args.agent_sizes.split(",")]
    if args.no_robustness:
        agent_sizes = [n_agents_primary]
    else:
        agent_sizes = sorted(set([n_agents_primary] + agent_sizes_raw))

    seeds = _parse_seeds_arg(args.seeds, n_seeds)
    n_seeds = len(seeds)

    clusters = load_expanded_clusters(_BENCHMARKS_PATH)

    print("=" * 68)
    print("  Expanded Cross-Cultural ESS Validation (Phase 27+)")
    print(f"  Clusters: {len(clusters)}   Seeds: {n_seeds}   Rounds: {n_rounds}")
    print(f"  Policy:   {policy_type}   Mode: {'dry-run' if dry_run else 'real'}")
    print(f"  Agent sizes: {agent_sizes}")
    print("=" * 68)

    # Load ESS parquet once
    ess_df = None
    if not dry_run:
        import pandas as pd

        ess_df = pd.read_parquet(_ESS_PATH)
        _assert_no_wvs_columns(ess_df)
        # Print cluster cohort sizes for transparency
        for c in clusters:
            n = ((ess_df["trust_people"] >= c.trust_lo) & (ess_df["trust_people"] < c.trust_hi)).sum()
            print(f"  [{c.name}] trust=[{c.trust_lo},{c.trust_hi})  AT cohort: n={n}")
        print()

    def _run_condition(label: str, grounded: bool) -> dict[int, list[ClusterMultiSeedResult]]:
        multi_results_by_size: dict[int, list[ClusterMultiSeedResult]] = {}
        for sz in agent_sizes:
            is_primary = sz == n_agents_primary
            print(f"\n── {label} | Agent size: {sz} ({'primary' if is_primary else 'robustness'}) ──")
            size_results: list[ClusterMultiSeedResult] = []

            for cluster in clusters:
                t0 = time.time()
                print(f"  [{cluster.name}] ESS trust={cluster.ess_mean_trust:.3f}", end="  ")
                result = run_cluster_multiseed(
                    cluster=cluster,
                    n_agents=sz,
                    n_rounds=n_rounds,
                    policy_type=policy_type,
                    seeds=seeds,
                    ess_df=ess_df,
                    dry_run=dry_run,
                    grounded=grounded,
                    verbose=True,
                )
                elapsed = time.time() - t0
                print(
                    f"    → mean_coop={result.mean_cooperation_rate:.3f} "
                    f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}] "
                    f"({elapsed:.0f}s)"
                )
                size_results.append(result)

            cc = compute_cross_cultural_correlation_multiseed(size_results)
            ess_fit = compute_benchmark_fit(size_results, benchmark="ess")
            wvs_fit = compute_benchmark_fit(size_results, benchmark="wvs")
            print(f"\n  ESS fit: Pearson r = {cc.pearson_r:+.3f}, Spearman ρ = {cc.spearman_rho:+.3f}")
            print(
                f"  WVS holdout: Pearson r = {wvs_fit.pearson_r:+.3f}, "
                f"Spearman ρ = {wvs_fit.spearman_rho:+.3f}, RMSE = {wvs_fit.rmse:.3f}"
            )
            print(f"  Gradient recovered: {'YES ✓' if cc.gradient_recovered else 'NO ✗'}")
            multi_results_by_size[sz] = size_results
        return multi_results_by_size

    grounded_results_by_size = _run_condition("Grounded", grounded=True)
    control_results_by_size = (
        _run_condition("Ungrounded control", grounded=False) if args.run_ungrounded_control else None
    )

    # Save results
    _save_expanded(
        _RESULTS_PATH,
        grounded_results_by_size,
        n_rounds,
        n_seeds,
        control_results_by_size=control_results_by_size,
        seeds=seeds,
        policy_type=policy_type,
        dry_run=dry_run,
        benchmarks_path=_BENCHMARKS_PATH,
        ess_path=_ESS_PATH,
        ess_row_count=None if ess_df is None else int(len(ess_df)),
    )
    print(f"\nResults saved to: {_RESULTS_PATH}")

    primary_results = grounded_results_by_size[n_agents_primary]
    cc_primary = compute_cross_cultural_correlation_multiseed(primary_results)
    wvs_primary = compute_benchmark_fit(primary_results, benchmark="wvs")
    _save_csv(_TABLE_PATH, primary_results, cc_primary, wvs_primary)
    print(f"Table saved to:   {_TABLE_PATH}")

    if control_results_by_size is not None:
        primary_control = control_results_by_size[n_agents_primary]
        holdout_cmp = compare_holdout_fit(primary_results, primary_control, benchmark="wvs")
        print("\nHoldout comparison (Grounded vs Ungrounded control, WVS benchmark):")
        print(f"  ΔPearson r   = {holdout_cmp.delta_pearson_r:+.3f}")
        print(f"  ΔSpearman ρ  = {holdout_cmp.delta_spearman_rho:+.3f}")
        print(f"  ΔMAE         = {holdout_cmp.delta_mae:+.3f}")
        print(f"  ΔRMSE        = {holdout_cmp.delta_rmse:+.3f}")
        print(f"  Grounded better: {'YES ✓' if holdout_cmp.grounded_better else 'NO ✗'}")
    else:
        print(
            "\n[warning] Ungrounded control was not run. "
            "For publication-grade external validation, rerun with --run-ungrounded-control."
        )

    # Print summary
    print()
    from metrics.cross_cultural import ClusterSimResult

    mock_singles = [
        ClusterSimResult(
            r.cluster_name,
            r.ess_mean_trust,
            r.mean_cooperation_rate,
            r.mean_gini,
            r.n_agents,
            r.n_rounds,
        )
        for r in primary_results
    ]
    from metrics.cross_cultural import compute_cross_cultural_correlation

    fmt_result = compute_cross_cultural_correlation(mock_singles)
    print(format_cross_cultural_table(fmt_result))
    print(
        "WVS holdout fit: "
        f"r={wvs_primary.pearson_r:+.3f}, "
        f"ρ={wvs_primary.spearman_rho:+.3f}, "
        f"RMSE={wvs_primary.rmse:.3f}"
    )

    print(f"\n  95% CI widths: {[round(r.ci_upper - r.ci_lower, 3) for r in primary_results]}")
    print(f"  N seeds: {n_seeds}   N clusters: {len(clusters)}")


if __name__ == "__main__":
    main()
