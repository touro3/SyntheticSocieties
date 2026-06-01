"""Long-Horizon Persona Drift Analysis.5.

Runs T=100 round simulations for two conditions (grounded vs ungrounded)
using synthetic agent profiles (no GPU required) and tracks persona fidelity
every 10 rounds using the metrics/persona_decay.py module.

Grounded (Condition B proxy):
  coop_prob = clip(0.2 + 0.5 * trust * (1 - risk) + 0.15 * social, 0.05, 0.90)
  Agents follow ESS-derived cooperation probabilities → fidelity stays high.

Ungrounded (Condition A proxy):
  coop_prob = 0.70 (flat — RLHF cooperative bias)
  All agents cooperate at ~70% regardless of trust → fidelity degrades for
  heterogeneous populations where expected cooperation spans 0.20–0.80.

Saves results to analysis/tables/long_horizon_persona_drift.json.

Usage
-----
    python scripts/run_long_horizon.py
    python scripts/run_long_horizon.py --n-agents 200 --n-rounds 100 --n-seeds 5
    python scripts/run_long_horizon.py --plot-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────


def _hash_uniform(agent_id: str, round_id: int) -> float:
    key = f"{agent_id}:{round_id}".encode()
    digest = hashlib.sha256(key).digest()
    return struct.unpack(">I", digest[:4])[0] / 4_294_967_296.0


def _expected_coop(trust: float, risk: float) -> float:
    """Persona-defined expected cooperation rate (mirrors persona_decay.py)."""
    return 0.2 + 0.6 * trust * (1.0 - risk)


def _fidelity(actual: float, expected: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - abs(actual - expected))))


# ── simulation ─────────────────────────────────────────────────────────────


def _run_condition(
    condition: str,
    n_agents: int,
    n_rounds: int,
    seed: int,
    window: int = 5,
) -> dict:
    """Simulate one condition (grounded or ungrounded) and compute per-round fidelity.

    Returns a dict with:
        rounds: list of round indices
        fidelity_mean: mean fidelity across agents, per round
        fidelity_std: std across agents, per round
        decay_rate: OLS slope of mean fidelity vs round
    """
    rng = np.random.default_rng(seed)
    n = n_agents

    trust = rng.beta(2, 2, size=n)
    risk = rng.beta(2, 3, size=n)
    social = rng.beta(2, 2, size=n)
    agent_ids = [f"agent_{i:04d}" for i in range(n)]

    expected = np.array([_expected_coop(float(trust[i]), float(risk[i])) for i in range(n)])

    # Sliding window: track cooperation in last `window` rounds per agent
    action_history: list[list[int]] = [[] for _ in range(n)]  # 1=cooperate, 0=not

    rounds_out: list[int] = []
    fidelity_mean_out: list[float] = []
    fidelity_std_out: list[float] = []

    for r in range(n_rounds):
        if condition == "grounded":
            coop_prob = np.clip(
                0.2 + 0.5 * trust * (1.0 - risk) + 0.15 * social,
                0.05,
                0.90,
            )
        else:  # ungrounded: flat RLHF bias
            coop_prob = np.full(n, 0.70)

        # Record actions
        for i in range(n):
            cooperated = 1 if _hash_uniform(agent_ids[i], r) < coop_prob[i] else 0
            action_history[i].append(cooperated)
            # Trim to window
            if len(action_history[i]) > window:
                action_history[i] = action_history[i][-window:]

        # Compute fidelity from windowed actual cooperation rate
        if r >= window - 1:  # enough history
            fidels = []
            for i in range(n):
                actual = float(np.mean(action_history[i]))
                fidels.append(_fidelity(actual, float(expected[i])))
            rounds_out.append(r)
            fidelity_mean_out.append(float(np.mean(fidels)))
            fidelity_std_out.append(float(np.std(fidels)))

    # OLS decay rate
    decay_rate = 0.0
    if len(rounds_out) >= 2:
        x = np.array(rounds_out, dtype=float)
        y = np.array(fidelity_mean_out, dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        var_x = ((x - x_mean) ** 2).sum()
        if var_x > 0:
            decay_rate = float(((x - x_mean) * (y - y_mean)).sum() / var_x)

    return {
        "condition": condition,
        "seed": seed,
        "rounds": rounds_out,
        "fidelity_mean": [round(v, 4) for v in fidelity_mean_out],
        "fidelity_std": [round(v, 4) for v in fidelity_std_out],
        "decay_rate": round(decay_rate, 6),
    }


# ── aggregate ──────────────────────────────────────────────────────────────


def _aggregate_across_seeds(runs: list[dict]) -> dict:
    """Average fidelity curves across seeds (assumes same rounds list)."""
    if not runs:
        return {}
    rounds = runs[0]["rounds"]
    n_rounds_valid = len(rounds)
    means = np.array([r["fidelity_mean"] for r in runs])  # shape (n_seeds, T)
    stds = np.array([r["fidelity_std"] for r in runs])

    decay_rates = [r["decay_rate"] for r in runs]

    return {
        "condition": runs[0]["condition"],
        "rounds": rounds,
        "fidelity_mean": [round(float(v), 4) for v in means.mean(axis=0)],
        "fidelity_ci_lower": [round(float(v), 4) for v in (means.mean(axis=0) - means.std(axis=0))],
        "fidelity_ci_upper": [round(float(v), 4) for v in (means.mean(axis=0) + means.std(axis=0))],
        "decay_rate_mean": round(float(np.mean(decay_rates)), 6),
        "decay_rate_std": round(float(np.std(decay_rates)), 6),
        "n_seeds": len(runs),
    }


# ── main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-horizon persona drift analysis")
    parser.add_argument("--n-agents", type=int, default=150)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--window", type=int, default=5, help="Sliding window for per-agent cooperation rate")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation — re-plot from existing JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path("analysis/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "long_horizon_persona_drift.json"

    if not args.plot_only:
        seeds = list(range(42, 42 + args.n_seeds))
        conditions = ["grounded", "ungrounded"]
        runs_by_condition: dict[str, list[dict]] = {c: [] for c in conditions}

        print(f"[long_horizon] Running T={args.n_rounds}, N={args.n_agents}, seeds={seeds}")
        for condition in conditions:
            for seed in seeds:
                print(f"  condition={condition}, seed={seed}…", end=" ", flush=True)
                result = _run_condition(
                    condition=condition,
                    n_agents=args.n_agents,
                    n_rounds=args.n_rounds,
                    seed=seed,
                    window=args.window,
                )
                runs_by_condition[condition].append(result)
                dr = result["decay_rate"]
                final_f = result["fidelity_mean"][-1] if result["fidelity_mean"] else float("nan")
                print(f"done (decay_rate={dr:+.5f}, final_fidelity={final_f:.3f})")

        aggregated = {c: _aggregate_across_seeds(runs_by_condition[c]) for c in conditions}

        print("\n[long_horizon] Summary:")
        for c, agg in aggregated.items():
            print(
                f"  {c:12s}  decay_rate={agg['decay_rate_mean']:+.5f} ± {agg['decay_rate_std']:.5f}"
                f"  final_fidelity={agg['fidelity_mean'][-1]:.3f}"
            )

        payload = {
            "n_agents": args.n_agents,
            "n_rounds": args.n_rounds,
            "n_seeds": args.n_seeds,
            "window": args.window,
            "results": aggregated,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[long_horizon] Saved to {json_path}")

    if json_path.exists():
        print("[long_horizon] Generating figures…")
        subprocess.run(
            [sys.executable, "scripts/plot_long_horizon.py", "--input", str(json_path)],
            check=False,
        )
    else:
        print(f"[long_horizon] JSON not found at {json_path}; skipping plot.")


if __name__ == "__main__":
    main()
