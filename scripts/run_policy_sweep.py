"""End-to-end policy parameter sweep for the Behavioral Grounding Framework.

Exercises InterventionEngine across a parameter grid and produces:
  - analysis/policy_sweep_results.json
  - analysis/figures/policy_sweep.pdf

Usage
-----
    python scripts/run_policy_sweep.py                          # redistribute_top_pct
    python scripts/run_policy_sweep.py --rule cooperation_bonus
    python scripts/run_policy_sweep.py --n-agents 20 --n-rounds 12
    python scripts/run_policy_sweep.py --dry-run               # synthetic data, no simulation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.policy_intervention import (
    InterventionEngine,
    PolicyIntervention,
    SweepPoint,
    sensitivity_index,
)
from environment.world import World
from environment.world_state import WorldState
from metrics.inequality import gini_coefficient
from metrics.policy_sensitivity import ClusterOutcome, direction_recovery
from simulation.kernel import SimulationKernel

# ── Parameters ────────────────────────────────────────────────────────────────

SWEEP_PARAMS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_RULE = "redistribute_top_pct"
DEFAULT_N_AGENTS = 10
DEFAULT_N_ROUNDS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _coop_rate(agents: list) -> float:
    """Fraction of agents whose last action was 'cooperate'."""
    actions = [a.state.last_action for a in agents if a.state.last_action]
    return sum(1 for a in actions if a == "cooperate") / len(actions) if actions else 0.0


def _make_agents(n: int, rule: str) -> list[Agent]:
    """Build a fresh list of agents with pre-existing wealth inequality."""
    agents = []
    for i in range(n):
        profile = AgentProfile(
            agent_id=f"agent_{i:03d}",
            age=30 + (i % 40),
            income=float(30 + i * 8),
            education="secondary",
            occupation="worker",
            location="EU",
            political_preference="centrist",
            risk_tolerance=0.5,
            social_class="middle",
        )
        state = AgentState(wealth=30.0 + i * 8.0)
        memory = MemoryBuffer(max_items=10)
        policy = MockPolicy()
        agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))
    return agents


def _make_world_and_kernel(agents: list, tmp_log: str) -> tuple[World, SimulationKernel]:
    """Instantiate a World + SimulationKernel with no NetworkManager."""
    world_state = WorldState(round_id=0)
    institution_manager = InstitutionManager()
    world = World(state=world_state, institution_manager=institution_manager, network_manager=None)
    logger = EventLogger(output_path=tmp_log, overwrite=True)
    kernel = SimulationKernel(agents=agents, world=world, logger=logger)
    return world, kernel


# ── Sweep ─────────────────────────────────────────────────────────────────────


def run_sweep(
    rule: str = DEFAULT_RULE,
    parameters: Optional[list[float]] = None,
    n_agents: int = DEFAULT_N_AGENTS,
    n_rounds: int = DEFAULT_N_ROUNDS,
) -> list[SweepPoint]:
    """Run the policy parameter sweep and return a SweepPoint per parameter."""
    if parameters is None:
        parameters = SWEEP_PARAMS

    sweep_results: list[SweepPoint] = []

    for param in parameters:
        # Fresh agents and world for each parameter value
        agents = _make_agents(n_agents, rule)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            tmp_log = tf.name

        try:
            world, kernel = _make_world_and_kernel(agents, tmp_log)

            # Pre-intervention phase
            kernel.run(n_rounds)
            pre_gini = gini_coefficient([a.state.wealth for a in agents])
            pre_coop = _coop_rate(agents)

            # Set up intervention to fire at the current round_id
            trigger = world.state.round_id
            intervention = PolicyIntervention(
                trigger_round=trigger,
                rule=rule,
                parameter=param,
                label=f"{rule}_{param:.2f}",
            )
            engine = InterventionEngine([intervention])
            engine.apply(round_id=world.state.round_id, agents=agents)

            # Post-intervention phase
            kernel.run(n_rounds)
            post_gini = gini_coefficient([a.state.wealth for a in agents])
            post_coop = _coop_rate(agents)

        finally:
            try:
                os.unlink(tmp_log)
            except OSError:
                pass

        sweep_results.append(
            SweepPoint(
                parameter=param,
                label=f"{rule}_{param:.2f}",
                pre_gini=round(pre_gini, 6),
                post_gini=round(post_gini, 6),
                pre_coop_rate=round(pre_coop, 6),
                post_coop_rate=round(post_coop, 6),
            )
        )

    return sweep_results


def dry_run_sweep(
    rule: str = DEFAULT_RULE,
    parameters: Optional[list[float]] = None,
) -> list[SweepPoint]:
    """Return deterministic synthetic SweepPoints (no simulation)."""
    if parameters is None:
        parameters = SWEEP_PARAMS

    results = []
    for i, param in enumerate(parameters):
        # Monotonically decreasing Gini, slight coop increase
        results.append(
            SweepPoint(
                parameter=param,
                label=f"{rule}_{param:.2f}",
                pre_gini=0.40,
                post_gini=round(0.40 - param * 0.35, 6),
                pre_coop_rate=0.30,
                post_coop_rate=round(0.30 + param * 0.10, 6),
            )
        )
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────


def _print_table(sweep_results: list[SweepPoint], rule: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Policy Sweep: {rule}")
    print(f"{'=' * 72}")
    print(f"  {'Param':>6}  {'Pre Gini':>9}  {'Post Gini':>10}  {'ΔGini':>8}  {'ΔCoop':>8}")
    print(f"  {'-' * 60}")
    for sp in sweep_results:
        print(
            f"  {sp.parameter:>6.2f}  {sp.pre_gini:>9.4f}  {sp.post_gini:>10.4f}"
            f"  {sp.delta_gini:>+8.4f}  {sp.delta_coop:>+8.4f}"
        )
    si_gini = sensitivity_index(sweep_results, "gini")
    si_coop = sensitivity_index(sweep_results, "coop")
    print(f"\n  Sensitivity index (Gini): {si_gini:.4f}")
    print(f"  Sensitivity index (Coop): {si_coop:.4f}")
    print(f"{'=' * 72}\n")


def save_results(sweep_results: list[SweepPoint], rule: str, output_path: Path) -> None:
    """Serialize sweep results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule": rule,
        "sensitivity_index_gini": sensitivity_index(sweep_results, "gini"),
        "sensitivity_index_coop": sensitivity_index(sweep_results, "coop"),
        "sweep_points": [
            {
                "parameter": sp.parameter,
                "label": sp.label,
                "pre_gini": sp.pre_gini,
                "post_gini": sp.post_gini,
                "pre_coop_rate": sp.pre_coop_rate,
                "post_coop_rate": sp.post_coop_rate,
                "delta_gini": sp.delta_gini,
                "delta_coop": sp.delta_coop,
            }
            for sp in sweep_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved → {output_path}")


def plot_sweep(sweep_results: list[SweepPoint], rule: str, figure_path: Path) -> None:
    """Generate 2-panel matplotlib figure (Agg backend, no display needed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = [sp.parameter for sp in sweep_results]
    delta_gini = [sp.delta_gini for sp in sweep_results]
    delta_coop = [sp.delta_coop for sp in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(params, delta_gini, "o-", color="steelblue", linewidth=2, markersize=7)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Policy Parameter")
    ax1.set_ylabel("ΔGini (post − pre)")
    ax1.set_title(f"Gini Reduction — {rule}")

    ax2.plot(params, delta_coop, "s-", color="darkorange", linewidth=2, markersize=7)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Policy Parameter")
    ax2.set_ylabel("ΔCoop rate (post − pre)")
    ax2.set_title(f"Cooperation Change — {rule}")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved  → {figure_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BGF policy parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rule",
        default=DEFAULT_RULE,
        choices=["redistribute_top_pct", "wealth_floor", "cooperation_bonus", "tax_income"],
        help="Policy rule to sweep",
    )
    parser.add_argument("--n-agents", type=int, default=DEFAULT_N_AGENTS)
    parser.add_argument("--n-rounds", type=int, default=DEFAULT_N_ROUNDS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic deterministic data (no simulation, fast CI mode)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "analysis" / "policy_sweep_results.json"),
        help="Path for JSON output",
    )
    parser.add_argument(
        "--figure",
        default=str(ROOT / "analysis" / "figures" / "policy_sweep.pdf"),
        help="Path for PDF figure",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.dry_run:
        print("[dry-run] Generating synthetic sweep data (no simulation).")
        sweep_results = dry_run_sweep(rule=args.rule)
    else:
        print(f"[sweep] Running live sweep: rule={args.rule}, n_agents={args.n_agents}, n_rounds={args.n_rounds}")
        sweep_results = run_sweep(
            rule=args.rule,
            n_agents=args.n_agents,
            n_rounds=args.n_rounds,
        )

    _print_table(sweep_results, args.rule)
    save_results(sweep_results, args.rule, Path(args.output))
    plot_sweep(sweep_results, args.rule, Path(args.figure))

    # Direction recovery check
    policy_pairs = [
        (
            sp.parameter,
            ClusterOutcome(
                cluster_name="policy_sweep",
                simulated_gini=sp.post_gini,
                simulated_coop=sp.post_coop_rate,
            ),
        )
        for sp in sweep_results
    ]
    dr = direction_recovery([], policy_parameter_pairs=policy_pairs)
    for r in dr:
        status = "✓" if r.recovered else "✗"
        print(f"  Direction [{r.check}]: {status} (Δ={r.delta:.4f})")


if __name__ == "__main__":
    main()
