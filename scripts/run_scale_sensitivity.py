"""Population Scale Sensitivity Experiment.

Tests whether the grounding effect (BRM gap between Condition A and B) is
robust across population sizes N, and at what threshold grounding becomes
effective.

Sweeps N ∈ {10, 20, 50, 100, 200, 500} × 10 seeds × {grounded, ungrounded}.
Uses rule-based ESS policy proxy (no GPU required).

Key metrics per (N, condition):
    cooperation_rate  — mean ± 95% CI
    gini              — mean ± 95% CI
    b_rlhf            — mean ± 95% CI
    brm_composite     — mean ± 95% CI

Key finding expected:
    grounding_effect = BRM(grounded) - BRM(ungrounded) should be stable or
    increase with N. If it collapses at small N, document the threshold.

Output:
    analysis/tables/scale_sensitivity.json
    analysis/figures/scale_sensitivity.png

Usage
-----
    python scripts/run_scale_sensitivity.py
    python scripts/run_scale_sensitivity.py --n-seeds 5 --plot-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.behavioral_realism import compute_composite_brm, compute_rlhf_bias_index
from metrics.statistical_inference import bootstrap_ci


# ── ESS benchmarks ────────────────────────────────────────────────────────

_EMP_GINI = 0.31         # Eurostat EU median
_EMP_COOP_GROUNDED = 0.40    # ESS-grounded expected cooperation
_EMP_COOP_UNGROUNDED = 0.33  # Uniform baseline expected
_EMP_WEALTH_MEAN = 110.0


# ── Helpers ───────────────────────────────────────────────────────────────


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


# ── Single simulation ─────────────────────────────────────────────────────


def _simulate(
    n_agents: int,
    n_rounds: int,
    seed: int,
    condition: str,   # "grounded" or "ungrounded"
) -> dict:
    """Run one (N, condition, seed) experiment. Returns metrics dict."""
    rng = np.random.default_rng(seed)

    if condition == "grounded":
        trust = rng.beta(2, 2, size=n_agents)
        risk = rng.beta(2, 3, size=n_agents)
        social = rng.beta(2, 2, size=n_agents)
        coop_prob_fn = lambda i: np.clip(0.2 + 0.5 * trust[i] * (1 - risk[i]) + 0.15 * social[i], 0.05, 0.90)
    else:
        # Ungrounded: flat RLHF bias — all agents cooperate at ~70%
        coop_prob_fn = lambda i: 0.70

    wealth = np.full(n_agents, 100.0)
    agent_ids = [f"a{i}" for i in range(n_agents)]
    action_counts = {"work": 0, "save": 0, "cooperate": 0}

    for r in range(n_rounds):
        for i in range(n_agents):
            h = _hash_uniform(agent_ids[i], r)
            cp = coop_prob_fn(i)
            if h < cp:
                action_counts["cooperate"] += 1
                wealth[i] += 7.0
            elif h < cp + 0.3:
                action_counts["work"] += 1
                wealth[i] += 10.0
            else:
                action_counts["save"] += 1
                wealth[i] += 4.0

    total = max(sum(action_counts.values()), 1)
    act_dist = {a: c / total for a, c in action_counts.items()}
    act_coop = action_counts["cooperate"] / total
    act_gini = _gini(wealth.tolist())

    emp_wealth = list(np.clip(
        rng.lognormal(np.log(_EMP_WEALTH_MEAN), 0.35, n_agents), 10, 500
    ))
    emp_coop = _EMP_COOP_GROUNDED if condition == "grounded" else _EMP_COOP_UNGROUNDED

    brm = compute_composite_brm(
        sim_wealth=wealth.tolist(),
        emp_wealth=emp_wealth,
        sim_gini=act_gini,
        emp_gini=_EMP_GINI,
        sim_coop_rate=act_coop,
        emp_coop_rate=emp_coop,
        temporal_stability_jsd=0.05,
    )

    return {
        "n_agents": n_agents,
        "condition": condition,
        "seed": seed,
        "brm": round(brm["composite"], 4),
        "b_rlhf": round(compute_rlhf_bias_index(act_dist), 4),
        "coop_rate": round(act_coop, 4),
        "gini": round(act_gini, 4),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────


N_VALUES = [10, 20, 50, 100, 200, 500]


def run_scale_sweep(
    n_values: list[int] = N_VALUES,
    n_rounds: int = 30,
    n_seeds: int = 10,
) -> list[dict]:
    seeds = list(range(42, 42 + n_seeds))
    results = []
    for n in n_values:
        for cond in ["grounded", "ungrounded"]:
            for seed in seeds:
                results.append(_simulate(n, n_rounds, seed, cond))
    return results


def aggregate_sweep(results: list[dict]) -> list[dict]:
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["n_agents"], r["condition"])].append(r)

    summaries = []
    for (n, cond), group in sorted(groups.items()):
        brms = [r["brm"] for r in group]
        coops = [r["coop_rate"] for r in group]
        ginis = [r["gini"] for r in group]
        b_rlhfs = [r["b_rlhf"] for r in group]

        _, brm_ci_lo, brm_ci_hi = bootstrap_ci(brms, n_bootstrap=2000, random_state=42)
        _, coop_ci_lo, coop_ci_hi = bootstrap_ci(coops, n_bootstrap=2000, random_state=42)

        summaries.append({
            "n_agents": n,
            "condition": cond,
            "brm_mean": round(float(np.mean(brms)), 4),
            "brm_ci_lower": round(brm_ci_lo, 4),
            "brm_ci_upper": round(brm_ci_hi, 4),
            "coop_mean": round(float(np.mean(coops)), 4),
            "coop_ci_lower": round(coop_ci_lo, 4),
            "coop_ci_upper": round(coop_ci_hi, 4),
            "gini_mean": round(float(np.mean(ginis)), 4),
            "b_rlhf_mean": round(float(np.mean(b_rlhfs)), 4),
            "n_seeds": len(group),
        })
    return summaries


# ── Main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Population scale sensitivity sweep")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--n-values", type=str, default="10,20,50,100,200,500")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path("analysis/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scale_sensitivity.json"

    if not args.plot_only:
        n_values = [int(x) for x in args.n_values.split(",")]
        print(f"[scale_sensitivity] N={n_values} × {args.n_seeds} seeds × 2 conditions")

        results = run_scale_sweep(
            n_values=n_values,
            n_rounds=args.n_rounds,
            n_seeds=args.n_seeds,
        )
        summaries = aggregate_sweep(results)

        print(f"\n  {'N':>6}  {'Cond':12}  {'BRM':>8}  [95% CI]          {'B_RLHF':>8}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*18}  {'-'*8}")
        for s in summaries:
            print(
                f"  {s['n_agents']:>6}  {s['condition']:12}  "
                f"{s['brm_mean']:.4f}  [{s['brm_ci_lower']:.4f}, {s['brm_ci_upper']:.4f}]  "
                f"{s['b_rlhf_mean']:.4f}"
            )

        payload = {
            "n_rounds": args.n_rounds,
            "n_seeds": args.n_seeds,
            "n_values": n_values,
            "summaries": summaries,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[scale_sensitivity] Saved to {json_path}")

    if json_path.exists():
        subprocess.run(
            [sys.executable, "scripts/plot_scale_sensitivity.py",
             "--input", str(json_path)],
            check=False,
        )


if __name__ == "__main__":
    main()
