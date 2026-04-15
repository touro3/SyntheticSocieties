#!/usr/bin/env python3
"""Run trust-gradient sub-population validation.

Phase 17 — Trust-Gradient Sub-Population Validation.

For each of 4 ESS trust bands (low → very_high), ground a population using
SocietySpec.trust_people_band, run a short simulation with the rule-based policy
(no GPU required), and verify that simulated cooperation rates follow the same
rank order as ESS trust means.

Usage:
    python scripts/run_trust_gradient.py
    python scripts/run_trust_gradient.py --rounds 15 --agents 100 --seeds 42,123,7

Output:
    analysis/tables/trust_gradient.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.rule_based_ess_policy import RuleBasedESSPolicy
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from metrics.event_metrics import behavior_summary_from_events, load_events
from metrics.inequality import gini_coefficient
from metrics.trust_gradient import (
    TRUST_GROUPS,
    TrustGroup,
    compute_trust_gradient,
    compute_trust_recovery_correlation,
)
from population.ess_grounding import ESSGrounder
from population.society_spec import SocietySpec
from simulation.kernel import SimulationKernel


def _make_agent_from_profile_row(row, agent_id: str, rng: np.random.Generator) -> Agent:
    """Build a BGF Agent from a grounded ESS profile row."""
    profile = AgentProfile(
        agent_id=agent_id,
        age=int(row.get("age", 35)),
        income=float(row.get("income_decile", 5.0)) * 200,
        education=str(row.get("education_level", "secondary")),
        occupation=str(row.get("occupation_code", "worker")),
        location="urban",
        political_preference="center",
        risk_tolerance=float(row.get("risk_taking", 0.5)),
        social_class="middle",
        trust_people=max(0.0, min(1.0, float(row.get("trust_people", 0.5)))),
        competitiveness=max(0.0, min(1.0, float(row.get("competitiveness", 0.5)))),
        social_activity=max(0.0, min(1.0, float(row.get("social_activity", 0.5)))),
    )
    state = AgentState(
        wealth=float(rng.uniform(40, 80)),
        stress=float(rng.uniform(0.0, 0.3)),
        satisfaction=float(row.get("life_satisfaction", 0.6)),
    )
    return Agent(
        profile=profile,
        state=state,
        memory=MemoryBuffer(max_items=20),
        policy=RuleBasedESSPolicy(),
    )


def run_group_simulation(
    group: TrustGroup,
    n_agents: int,
    n_rounds: int,
    seed: int,
    ess_path: Path,
    tmp_dir: Path,
) -> dict:
    """Run a single trust-group simulation and return metrics.

    Returns:
        dict with keys: coop_rate, gini, mean_wealth, seed, group_name
    """
    rng = np.random.default_rng(seed)

    spec = SocietySpec(
        narrative=f"{group.name} sub-population grounded in ESS trust band '{group.band}'.",
        trust_people_band=group.band,
        target_population_size=n_agents,
    )

    grounder = ESSGrounder(ess_path=ess_path, min_cohort_size=30)
    result = grounder.ground(spec)
    cohort = result.matched_df

    if len(cohort) < n_agents:
        # Sample with replacement if cohort is too small
        cohort = cohort.sample(n=n_agents, replace=True, random_state=seed)
    else:
        cohort = cohort.sample(n=n_agents, replace=False, random_state=seed)

    agents = [
        _make_agent_from_profile_row(
            row=cohort.iloc[i].to_dict(),
            agent_id=f"{group.band}_s{seed}_a{i:04d}",
            rng=rng,
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

    log_path = tmp_dir / f"trust_{group.band}_s{seed}.jsonl"
    event_logger = EventLogger(log_path, overwrite=True)

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=event_logger,
    )
    kernel.run(num_rounds=n_rounds)

    events = load_events(log_path)
    behavior = behavior_summary_from_events(events)

    final_wealth = [a.state.wealth for a in agents]
    gini = gini_coefficient(final_wealth)
    mean_wealth = float(np.mean(final_wealth))
    coop_rate = float(behavior.get("event_behavior", {}).get("cooperation_rate", 0.0))

    return {
        "group_name": group.name,
        "band": group.band,
        "coop_rate": coop_rate,
        "gini": gini,
        "mean_wealth": mean_wealth,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "seed": seed,
        "cohort_size": len(cohort),
        "ess_reference_trust": group.ess_reference_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trust-gradient sub-population validation")
    parser.add_argument("--rounds", type=int, default=10, help="Simulation rounds per group")
    parser.add_argument("--agents", type=int, default=60, help="Agents per group")
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated seeds")
    parser.add_argument("--ess-path", type=str, default="data/ess_clean.parquet")
    parser.add_argument("--output", type=str, default="analysis/tables/trust_gradient.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ess_path = PROJECT_ROOT / args.ess_path
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = PROJECT_ROOT / "experiments" / "_trust_gradient_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Trust-Gradient Sub-Population Validation")
    print(f"  Groups: {len(TRUST_GROUPS)}, Rounds: {args.rounds}, Agents: {args.agents}, Seeds: {seeds}")
    print(f"{'=' * 60}\n")

    # Collect per-seed results for each group
    all_runs: list[dict] = []

    for group in TRUST_GROUPS:
        seed_results = []
        for seed in seeds:
            print(f"  [{group.name}] seed={seed} ...", end=" ", flush=True)
            try:
                run = run_group_simulation(
                    group=group,
                    n_agents=args.agents,
                    n_rounds=args.rounds,
                    seed=seed,
                    ess_path=ess_path,
                    tmp_dir=tmp_dir,
                )
                seed_results.append(run)
                print(f"coop_rate={run['coop_rate']:.3f}  gini={run['gini']:.3f}")
            except Exception as exc:
                print(f"FAILED: {exc}")

        all_runs.extend(seed_results)

    # Aggregate: mean coop_rate across seeds per group
    group_results: dict[str, dict] = {}
    for group in TRUST_GROUPS:
        runs = [r for r in all_runs if r["group_name"] == group.name]
        if not runs:
            print(f"WARNING: No successful runs for {group.name}")
            group_results[group.name] = {"coop_rate": 0.0, "gini": 0.0, "mean_wealth": 0.0}
            continue

        group_results[group.name] = {
            "coop_rate": float(np.mean([r["coop_rate"] for r in runs])),
            "gini": float(np.mean([r["gini"] for r in runs])),
            "mean_wealth": float(np.mean([r["mean_wealth"] for r in runs])),
            "coop_rate_std": float(np.std([r["coop_rate"] for r in runs])),
            "n_seeds": len(runs),
            "ess_reference_trust": group.ess_reference_mean,
        }

    # Compute correlation
    gradient = compute_trust_gradient(group_results)
    correlation = compute_trust_recovery_correlation(group_results)

    # Print summary table
    print(f"\n{'─' * 60}")
    print(f"  {'Group':<20} {'ESS Trust':>10} {'Coop Rate':>10} {'Gini':>8}")
    print(f"{'─' * 60}")
    for group in TRUST_GROUPS:
        r = group_results[group.name]
        print(f"  {group.name:<20} {group.ess_reference_mean:>10.3f} {r['coop_rate']:>10.3f} {r['gini']:>8.3f}")
    print(f"{'─' * 60}")
    print(
        f"\n  Spearman r = {correlation['spearman_r']:.3f}  "
        f"(p = {correlation['p_value']:.3f}, n = {correlation['n_groups']})"
    )
    print(f"  {correlation['interpretation']}\n")

    # Save results
    output = {
        "group_results": group_results,
        "gradient": gradient,
        "correlation": correlation,
        "all_runs": all_runs,
        "config": {
            "rounds": args.rounds,
            "agents": args.agents,
            "seeds": seeds,
            "policy": "rule_based",
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {output_path}")


if __name__ == "__main__":
    main()
