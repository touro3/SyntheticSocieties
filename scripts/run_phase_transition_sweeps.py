#!/usr/bin/env python3
"""Run parameter sweeps for phase transition analysis.

Phase 18 — Emergent complexity analysis.

Runs self-contained simulations (no GPU, no subprocess) across three sweeps:
  1. Bad apple fraction:  0% to 40% in 2% steps
  2. Shock magnitude:     0% to 100% in 10% steps
  3. Network rewiring:    beta 0.0 to 1.0 in 0.1 steps

Uses a rule-based ESS proxy with hash-based determinism.

Usage:
    python scripts/run_phase_transition_sweeps.py
    python scripts/run_phase_transition_sweeps.py --n-seeds 5 --n-agents 100

Output:
    analysis/tables/phase_transitions.json
    analysis/figures/phase_transitions.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.complexity import analyze_sweep_results, fit_power_law


# ── Shared helpers ─────────────────────────────────────────────────────────


def _hash_uniform(agent_id: str, round_id: int) -> float:
    key = f"{agent_id}:{round_id}".encode()
    digest = hashlib.sha256(key).digest()
    return struct.unpack(">I", digest[:4])[0] / 4_294_967_296.0


def _gini(values: list[float]) -> float:
    arr = sorted(values)
    n, s = len(arr), sum(arr)
    if n == 0 or s == 0:
        return 0.0
    return (2.0 * sum((i + 1) * v for i, v in enumerate(arr))) / (n * s) - (n + 1.0) / n


def _ess_coop_prob(trust: float, risk: float) -> float:
    return max(0.05, min(0.90, 0.2 + 0.5 * trust * (1.0 - risk)))


# ── Sweep 1: Bad apple fraction ────────────────────────────────────────────


def simulate_bad_apple(
    bad_apple_fraction: float,
    n_agents: int = 100,
    n_rounds: int = 30,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    trust = rng.beta(2, 2, size=n_agents)
    risk = rng.beta(2, 3, size=n_agents)
    is_bad = np.arange(n_agents) < int(n_agents * bad_apple_fraction)

    wealth = np.full(n_agents, 100.0)
    action_counts = {"work": 0, "save": 0, "cooperate": 0, "steal": 0}

    for r in range(n_rounds):
        for i in range(n_agents):
            h = _hash_uniform(f"a{i}", r)
            if is_bad[i]:
                # Bad apple: steal (damages public pool / neighbours)
                action_counts["steal"] += 1
                wealth[i] += 15.0
                # Penalise random neighbour
                target = rng.integers(0, n_agents)
                if target != i:
                    wealth[target] = max(0.0, wealth[target] - 5.0)
            else:
                cp = _ess_coop_prob(float(trust[i]), float(risk[i]))
                if wealth[i] < 70.0:
                    action_counts["work"] += 1
                    wealth[i] += 10.0
                elif h < cp:
                    action_counts["cooperate"] += 1
                    wealth[i] -= 3.0
                    target = rng.integers(0, n_agents)
                    wealth[target] = min(200.0, wealth[target] + 4.5)
                elif h < cp + 0.4:
                    action_counts["save"] += 1
                else:
                    action_counts["work"] += 1
                    wealth[i] += 10.0

    total = max(sum(action_counts.values()), 1)
    non_steal = max(total - action_counts["steal"], 1)
    return {
        "cooperation_rate": action_counts["cooperate"] / non_steal,
        "gini": _gini(wealth.tolist()),
        "mean_wealth": float(np.mean(wealth)),
    }


# ── Sweep 2: Shock magnitude ───────────────────────────────────────────────


def simulate_shock(
    shock_magnitude: float,
    n_agents: int = 100,
    n_rounds: int = 30,
    shock_round: int = 15,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    trust = rng.beta(2, 2, size=n_agents)
    risk = rng.beta(2, 3, size=n_agents)

    wealth = np.full(n_agents, 100.0)
    action_counts = {"work": 0, "save": 0, "cooperate": 0}

    for r in range(n_rounds):
        # Apply shock at specified round
        if r == shock_round:
            wealth *= (1.0 - shock_magnitude)

        for i in range(n_agents):
            h = _hash_uniform(f"a{i}", r)
            cp = _ess_coop_prob(float(trust[i]), float(risk[i]))

            if wealth[i] < 70.0:
                action_counts["work"] += 1
                wealth[i] += 10.0
            elif h < cp:
                action_counts["cooperate"] += 1
                wealth[i] -= 3.0
                target = rng.integers(0, n_agents)
                wealth[target] = min(200.0, wealth[target] + 4.5)
            elif h < cp + 0.4:
                action_counts["save"] += 1
            else:
                action_counts["work"] += 1
                wealth[i] += 10.0

    total = max(sum(action_counts.values()), 1)
    return {
        "cooperation_rate": action_counts["cooperate"] / total,
        "gini": _gini(wealth.tolist()),
        "mean_wealth": float(np.mean(wealth)),
    }


# ── Sweep 3: Network rewiring beta ────────────────────────────────────────


def simulate_beta(
    rewiring_prob: float,
    n_agents: int = 100,
    n_rounds: int = 30,
    seed: int = 42,
) -> dict:
    """Vary network rewiring probability.

    Higher beta → more random topology → lower clustering → cooperation is less
    reinforced by local trust. We model this as a penalty on cooperation
    probability that grows with beta (strangers trust less).

    trust_penalty = 0.25 * rewiring_prob   (up to −25pp at beta=1.0)
    """
    rng = np.random.default_rng(seed)
    trust = rng.beta(2, 2, size=n_agents)
    risk = rng.beta(2, 3, size=n_agents)

    wealth = np.full(n_agents, 100.0)
    action_counts = {"work": 0, "save": 0, "cooperate": 0}

    # Clustering penalty: high beta means random neighbours → less trust
    trust_penalty = 0.25 * rewiring_prob

    for r in range(n_rounds):
        for i in range(n_agents):
            h = _hash_uniform(f"a{i}", r)
            cp = max(0.05, _ess_coop_prob(float(trust[i]), float(risk[i])) - trust_penalty)

            if wealth[i] < 70.0:
                action_counts["work"] += 1
                wealth[i] += 10.0
            elif h < cp:
                action_counts["cooperate"] += 1
                wealth[i] -= 3.0
                target = rng.integers(0, n_agents)
                wealth[target] = min(200.0, wealth[target] + 4.5)
            elif h < cp + 0.4:
                action_counts["save"] += 1
            else:
                action_counts["work"] += 1
                wealth[i] += 10.0

    total = max(sum(action_counts.values()), 1)
    return {
        "cooperation_rate": action_counts["cooperate"] / total,
        "gini": _gini(wealth.tolist()),
        "mean_wealth": float(np.mean(wealth)),
    }


# ── Multi-seed aggregation ─────────────────────────────────────────────────


def run_sweep_direct(
    sweep_fn,
    param_values: list[float],
    n_seeds: int,
    n_agents: int,
    n_rounds: int,
    sweep_name: str,
    **kwargs,
) -> dict[str, list[float]]:
    """Run a sweep function across param_values × n_seeds, return aggregated metrics."""
    seeds = list(range(42, 42 + n_seeds))
    coop_rates, ginis, mean_wealths = [], [], []

    for val in param_values:
        seed_coops, seed_ginis, seed_mw = [], [], []
        for seed in seeds:
            result = sweep_fn(val, n_agents=n_agents, n_rounds=n_rounds, seed=seed, **kwargs)
            seed_coops.append(result["cooperation_rate"])
            seed_ginis.append(result["gini"])
            seed_mw.append(result["mean_wealth"])

        coop_rates.append(float(np.mean(seed_coops)))
        ginis.append(float(np.mean(seed_ginis)))
        mean_wealths.append(float(np.mean(seed_mw)))

        print(
            f"  [{sweep_name}] val={val:.2f}  coop={coop_rates[-1]:.3f}  "
            f"gini={ginis[-1]:.3f}  mean_w={mean_wealths[-1]:.1f}"
        )

    return {
        "cooperation_rate": coop_rates,
        "gini": ginis,
        "mean_wealth": mean_wealths,
    }


# ── Main ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase transition sweep analysis")
    parser.add_argument("--n-agents", type=int, default=100)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / "analysis" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_transitions.json"

    if not args.plot_only:
        # ── Define sweeps ────────────────────────────────────────────────
        bad_apple_values = [round(i * 0.02, 4) for i in range(21)]
        shock_values = [round(i * 0.1, 4) for i in range(11)]
        beta_values = [round(i * 0.1, 4) for i in range(11)]

        all_results: dict[str, dict] = {}

        # Sweep 1: bad apple fraction
        print(f"\n[bad_apple] {len(bad_apple_values)} points × {args.n_seeds} seeds")
        metrics = run_sweep_direct(
            simulate_bad_apple, bad_apple_values,
            n_seeds=args.n_seeds, n_agents=args.n_agents,
            n_rounds=args.n_rounds, sweep_name="bad_apple",
        )
        valid_mask = ~np.isnan(metrics["cooperation_rate"])
        valid_x = np.array(bad_apple_values)[valid_mask]
        valid_m = {k: np.array(v)[valid_mask] for k, v in metrics.items()}
        analysis = analyze_sweep_results(valid_x, valid_m)
        all_results["bad_apple"] = {
            "sweep_values": valid_x.tolist(),
            "metrics": {k: v.tolist() for k, v in valid_m.items()},
            "analysis": {
                k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
                    for kk, vv in v.items()}
                for k, v in analysis.items()
            },
        }

        # Sweep 2: shock magnitude
        print(f"\n[shock] {len(shock_values)} points × {args.n_seeds} seeds")
        metrics = run_sweep_direct(
            simulate_shock, shock_values,
            n_seeds=args.n_seeds, n_agents=args.n_agents,
            n_rounds=args.n_rounds, sweep_name="shock",
        )
        valid_mask = ~np.isnan(metrics["cooperation_rate"])
        valid_x = np.array(shock_values)[valid_mask]
        valid_m = {k: np.array(v)[valid_mask] for k, v in metrics.items()}
        analysis = analyze_sweep_results(valid_x, valid_m)
        all_results["shock"] = {
            "sweep_values": valid_x.tolist(),
            "metrics": {k: v.tolist() for k, v in valid_m.items()},
            "analysis": {
                k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
                    for kk, vv in v.items()}
                for k, v in analysis.items()
            },
        }

        # Sweep 3: network rewiring
        print(f"\n[beta] {len(beta_values)} points × {args.n_seeds} seeds")
        metrics = run_sweep_direct(
            simulate_beta, beta_values,
            n_seeds=args.n_seeds, n_agents=args.n_agents,
            n_rounds=args.n_rounds, sweep_name="beta",
        )
        valid_mask = ~np.isnan(metrics["cooperation_rate"])
        valid_x = np.array(beta_values)[valid_mask]
        valid_m = {k: np.array(v)[valid_mask] for k, v in metrics.items()}
        analysis = analyze_sweep_results(valid_x, valid_m)
        all_results["beta"] = {
            "sweep_values": valid_x.tolist(),
            "metrics": {k: v.tolist() for k, v in valid_m.items()},
            "analysis": {
                k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
                    for kk, vv in v.items()}
                for k, v in analysis.items()
            },
        }

        # Summary
        print(f"\n{'─'*60}")
        for sweep_name, result in all_results.items():
            for metric_name, a in result["analysis"].items():
                status = "TRANSITION" if a["is_transition"] else "no transition"
                print(
                    f"  [{sweep_name}/{metric_name}] {status} "
                    f"(inflection={a['inflection_point']:.3f}, "
                    f"R²={a['r_squared']:.3f})"
                )

        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=_json_safe)
        print(f"\nResults saved to {output_path}")

    # Plot
    if output_path.exists():
        subprocess.run(
            [sys.executable, "scripts/plot_phase_transitions.py",
             "--input", str(output_path)],
            check=False, cwd=str(PROJECT_ROOT),
        )


if __name__ == "__main__":
    main()
