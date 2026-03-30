"""Bad apple fraction sweep — empirically measure f* (phase transition inflection).

Sweeps adversarial agent injection fractions from 0% to 40% and fits a sigmoid
to the cooperation-vs-fraction curve to extract:
  - f*: inflection point (where cooperation drops fastest)
  - k:  steepness
  - R²: goodness-of-fit

Outputs:
  analysis/bad_apple_sweep.json
  analysis/figures/bad_apple_sweep.pdf

Usage
-----
    python scripts/run_bad_apple_sweep.py
    python scripts/run_bad_apple_sweep.py --n-agents 30 --n-rounds 20 --n-seeds 3
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
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from metrics.inequality import gini_coefficient
from simulation.kernel import SimulationKernel


class _WorkOnlyPolicy:
    """Adversarial agent policy: always work, never cooperate."""

    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="Adversarial agent works without cooperating.",
            confidence=1.0,
        )


FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_N_AGENTS = 20
DEFAULT_N_ROUNDS = 20
DEFAULT_N_SEEDS = 3


# ── Agent factory ─────────────────────────────────────────────────────────────

def _make_agents(n: int, bad_fraction: float, seed: int) -> list[Agent]:
    rng = np.random.default_rng(seed)
    n_bad = int(round(n * bad_fraction))
    bad_indices = set(rng.choice(n, size=n_bad, replace=False).tolist()) if n_bad > 0 else set()

    agents = []
    for i in range(n):
        is_bad = i in bad_indices
        profile = AgentProfile(
            agent_id=f"agent_{i:03d}",
            age=int(rng.integers(20, 70)),
            income=float(30 + i * 5),
            education="secondary",
            occupation="worker",
            location="EU",
            political_preference="centrist",
            risk_tolerance=float(rng.uniform(0.3, 0.7)),
            social_class="middle",
        )
        # Start above cooperation threshold (100) so normal agents cooperate
        # immediately. Adversarial non-cooperators break reciprocity, draining
        # wealth of normal neighbours below the threshold over time.
        state = AgentState(wealth=110.0)
        memory = MemoryBuffer(max_items=10)
        # Adversarial agents never cooperate; normal agents use RuleBasedPolicy
        # (cooperates when wealth >= 100 and has neighbours).
        policy = _WorkOnlyPolicy() if is_bad else RuleBasedPolicy()
        agent = Agent(profile=profile, state=state, memory=memory, policy=policy)
        agent.is_adversarial = is_bad
        agents.append(agent)
    return agents


def _run_one(bad_fraction: float, seed: int, n_agents: int, n_rounds: int) -> dict:
    agents = _make_agents(n_agents, bad_fraction, seed)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        tmp_log = tf.name

    try:
        agent_ids = [a.profile.agent_id for a in agents]
        world_state = WorldState(round_id=0)
        world = World(
            state=world_state,
            institution_manager=InstitutionManager(),
            network_manager=NetworkManager.small_world(agent_ids, k=4, rewiring_prob=0.1, seed=seed),
        )
        logger = EventLogger(output_path=tmp_log, overwrite=True)
        kernel = SimulationKernel(agents=agents, world=world, logger=logger)
        kernel.run(n_rounds)
    finally:
        try:
            os.unlink(tmp_log)
        except OSError:
            pass

    non_bad = [a for a in agents if not getattr(a, "is_adversarial", False)]
    actions = [a.state.last_action for a in non_bad if a.state.last_action]
    coop = sum(1 for a in actions if a == "cooperate") / len(actions) if actions else 0.0
    gini = gini_coefficient([a.state.wealth for a in agents])

    return {
        "bad_fraction": bad_fraction,
        "seed": seed,
        "coop_rate": round(coop, 6),
        "gini": round(gini, 6),
        "n_bad": int(round(n_agents * bad_fraction)),
    }


# ── Sigmoid fitting ───────────────────────────────────────────────────────────

def _fit_sigmoid(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit L / (1 + exp(-k*(x - f*))) to data. Returns params + R²."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {"note": "scipy not installed — sigmoid fit skipped"}

    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    try:
        # y is decreasing: flip to use increasing sigmoid, then flip back
        y_flip = 1 - y
        p0 = [max(y_flip), -10, 0.2]
        popt, _ = curve_fit(sigmoid, x, y_flip, p0=p0, maxfev=10000)
        L, k, f_star = popt

        y_pred = 1 - sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "L": round(float(L), 6),
            "k": round(float(abs(k)), 6),
            "f_star": round(float(f_star), 6),
            "r_squared": round(float(r2), 6),
        }
    except Exception as e:
        return {"note": f"sigmoid fit failed: {e}"}


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    fractions=None,
    n_agents: int = DEFAULT_N_AGENTS,
    n_rounds: int = DEFAULT_N_ROUNDS,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> dict:
    if fractions is None:
        fractions = FRACTIONS

    seeds = list(range(42, 42 + n_seeds))
    per_fraction = []

    for frac in fractions:
        seed_coops = []
        seed_ginis = []
        for seed in seeds:
            r = _run_one(frac, seed, n_agents, n_rounds)
            seed_coops.append(r["coop_rate"])
            seed_ginis.append(r["gini"])

        per_fraction.append({
            "bad_fraction": frac,
            "coop_rate_mean": round(float(np.mean(seed_coops)), 6),
            "coop_rate_std": round(float(np.std(seed_coops)), 6),
            "gini_mean": round(float(np.mean(seed_ginis)), 6),
            "gini_std": round(float(np.std(seed_ginis)), 6),
            "n_seeds": n_seeds,
        })
        print(f"  f={frac:.2f}: coop={np.mean(seed_coops):.3f}±{np.std(seed_coops):.3f}  "
              f"gini={np.mean(seed_ginis):.3f}")

    # Sigmoid fit
    fracs_arr = np.array([r["bad_fraction"] for r in per_fraction])
    coops_arr = np.array([r["coop_rate_mean"] for r in per_fraction])
    fit = _fit_sigmoid(fracs_arr, coops_arr)

    return {
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "n_seeds": n_seeds,
        "per_fraction": per_fraction,
        "sigmoid_fit": fit,
        "paper_claim": {"f_star": 0.12, "k": 18.2, "r_squared": 0.94},
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_sweep(results: dict, figure_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = [r["bad_fraction"] for r in results["per_fraction"]]
    coops = [r["coop_rate_mean"] for r in results["per_fraction"]]
    coop_errs = [r["coop_rate_std"] for r in results["per_fraction"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(fracs, coops, yerr=coop_errs, fmt="o-", color="steelblue",
                linewidth=2, markersize=7, capsize=4, label="Coop rate (mean ± std)")

    # Plot sigmoid fit if available
    fit = results.get("sigmoid_fit", {})
    if "f_star" in fit and "k" in fit and "L" in fit:
        x_plot = np.linspace(0, 0.4, 200)
        L, k, f_star = fit["L"], fit["k"], fit["f_star"]
        y_fit = 1 - (L / (1 + np.exp(-k * (x_plot - f_star))))
        ax.plot(x_plot, y_fit, "--", color="tomato", linewidth=1.5,
                label=f"Sigmoid fit: f*={f_star:.2f}, k={k:.1f}, R²={fit['r_squared']:.3f}")
        ax.axvline(f_star, color="tomato", alpha=0.4, linewidth=1)

    ax.set_xlabel("Bad apple fraction $f$")
    ax.set_ylabel("Cooperation rate (non-adversarial agents)")
    ax.set_title("Bad Apple Phase Transition Sweep")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.01, 0.42)
    ax.set_ylim(-0.05, 1.05)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {figure_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bad apple fraction sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-agents", type=int, default=DEFAULT_N_AGENTS)
    parser.add_argument("--n-rounds", type=int, default=DEFAULT_N_ROUNDS)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--out", default=str(ROOT / "analysis" / "bad_apple_sweep.json"))
    parser.add_argument("--figure", default=str(ROOT / "analysis" / "figures" / "bad_apple_sweep.pdf"))
    args = parser.parse_args(argv)

    print(f"Running bad apple sweep: fractions={FRACTIONS}")
    print(f"  n_agents={args.n_agents}, n_rounds={args.n_rounds}, n_seeds={args.n_seeds}")

    results = run_sweep(
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        n_seeds=args.n_seeds,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")

    fit = results.get("sigmoid_fit", {})
    print("\n  Sigmoid fit:")
    for k, v in fit.items():
        print(f"    {k}: {v}")
    claim = results.get("paper_claim", {})
    print(f"\n  Paper claims: f*={claim.get('f_star')}, k={claim.get('k')}, R²={claim.get('r_squared')}")

    try:
        plot_sweep(results, Path(args.figure))
    except Exception as e:
        print(f"  [warn] Figure generation failed: {e}")


if __name__ == "__main__":
    main()
