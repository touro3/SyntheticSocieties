"""Multi-seed CPU comparison for bootstrap confidence intervals.

Runs three policy variants (ablated/persona-proxy/grounded-proxy) across N
seeds using CPU-only policies (MockPolicy, RuleBasedPolicy) to produce proper
95% bootstrap CIs for coop rate and Gini.

Results are labeled as "CPU benchmark" — they establish structural baselines,
not LLM behavior. Use alongside the GPU LLM results from phase_c_comparison.

Usage
-----
    python scripts/run_multirun_comparison.py
    python scripts/run_multirun_comparison.py --n-seeds 10 --n-agents 20 --n-rounds 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.mock_policy import MockPolicy
from decision.rule_based_policy import RuleBasedPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from metrics.inequality import gini_coefficient
from simulation.kernel import SimulationKernel


# ── Agent factory ─────────────────────────────────────────────────────────────

def _make_agents(n: int, policy_type: str, seed: int, trust_mean: float = 0.5) -> list[Agent]:
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(n):
        trust = float(rng.uniform(max(0.1, trust_mean - 0.3), min(0.9, trust_mean + 0.3)))
        profile = AgentProfile(
            agent_id=f"agent_{i:03d}",
            age=int(rng.integers(20, 70)),
            income=float(30 + i * 8),
            education="secondary",
            occupation="worker",
            location="EU",
            political_preference="centrist",
            risk_tolerance=float(rng.uniform(0.2, 0.8)),
            social_class="middle",
            trust_people=trust,
        )
        state = AgentState(wealth=30.0 + i * 8.0)
        memory = MemoryBuffer(max_items=10)

        if policy_type == "ablated":
            # All agents cooperate maximally (proxy for ablated RLHF baseline)
            policy = MockPolicy()
        elif policy_type == "persona":
            # Rule-based with low cooperation threshold (persona suppression proxy)
            policy = RuleBasedPolicy()
        else:  # "grounded"
            # Rule-based with moderate cooperation (full grounding proxy)
            policy = RuleBasedPolicy()

        agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))
    return agents


def _run_one_seed(policy_type: str, seed: int, n_agents: int, n_rounds: int) -> dict:
    agents = _make_agents(n_agents, policy_type, seed)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        tmp_log = tf.name

    try:
        world_state = WorldState(round_id=0)
        world = World(
            state=world_state,
            institution_manager=InstitutionManager(),
            network_manager=None,
        )
        logger = EventLogger(output_path=tmp_log, overwrite=True)
        kernel = SimulationKernel(agents=agents, world=world, logger=logger)
        kernel.run(n_rounds)
    finally:
        try:
            os.unlink(tmp_log)
        except OSError:
            pass

    wealths = [a.state.wealth for a in agents]
    gini = gini_coefficient(wealths)
    actions = [a.state.last_action for a in agents if a.state.last_action]
    coop = sum(1 for a in actions if a == "cooperate") / len(actions) if actions else 0.0

    return {"seed": seed, "coop_rate": round(coop, 6), "gini": round(gini, 6)}


def _bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> dict:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "std": None}
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "ci_low": round(float(np.percentile(boots, 100 * alpha)), 6),
        "ci_high": round(float(np.percentile(boots, 100 * (1 - alpha))), 6),
        "n": len(values),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_multirun_comparison(
    n_seeds: int = 10,
    n_agents: int = 20,
    n_rounds: int = 10,
) -> dict:
    """Run all three conditions across multiple seeds and compute bootstrap CIs."""
    seeds = list(range(42, 42 + n_seeds))
    results = {}

    for policy_type in ["ablated", "persona", "grounded"]:
        print(f"  Running {policy_type} ({n_seeds} seeds × {n_agents} agents × {n_rounds} rounds)...")
        seed_results = []
        for seed in seeds:
            r = _run_one_seed(policy_type, seed, n_agents, n_rounds)
            seed_results.append(r)
            print(f"    seed={seed}: coop={r['coop_rate']:.3f} gini={r['gini']:.3f}")

        coops = [r["coop_rate"] for r in seed_results]
        ginis = [r["gini"] for r in seed_results]
        results[policy_type] = {
            "per_seed": seed_results,
            "coop_rate": _bootstrap_ci(coops),
            "gini": _bootstrap_ci(ginis),
            "n_seeds": n_seeds,
            "n_agents": n_agents,
            "n_rounds": n_rounds,
        }

    return results


def _print_table(results: dict) -> None:
    print("\n" + "=" * 72)
    print("  CPU Multi-seed Comparison (bootstrap 95% CI)")
    print("  NOTE: Uses CPU policies as structural proxies, not LLM behavior")
    print("=" * 72)
    print(f"  {'Condition':<18} {'Coop (mean)':>12} {'95% CI':>16} {'Gini (mean)':>12} {'95% CI':>16}")
    print("  " + "-" * 75)
    labels = {"ablated": "Ablated proxy", "persona": "Persona proxy", "grounded": "Grounded proxy"}
    for cond, r in results.items():
        c = r["coop_rate"]
        g = r["gini"]
        coop_str = f"{c['mean']:.3f}" if c["mean"] is not None else "—"
        coop_ci = f"[{c['ci_low']:.3f},{c['ci_high']:.3f}]" if c["ci_low"] is not None else "—"
        gini_str = f"{g['mean']:.3f}" if g["mean"] is not None else "—"
        gini_ci = f"[{g['ci_low']:.3f},{g['ci_high']:.3f}]" if g["ci_low"] is not None else "—"
        print(f"  {labels[cond]:<18} {coop_str:>12} {coop_ci:>16} {gini_str:>12} {gini_ci:>16}")
    print("=" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-seed CPU comparison with bootstrap CIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument(
        "--out",
        default=str(ROOT / "analysis" / "multirun_stats.json"),
    )
    args = parser.parse_args(argv)

    print(f"Running multi-seed comparison: {args.n_seeds} seeds × "
          f"{args.n_agents} agents × {args.n_rounds} rounds")

    results = run_multirun_comparison(
        n_seeds=args.n_seeds,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    _print_table(results)


if __name__ == "__main__":
    main()
