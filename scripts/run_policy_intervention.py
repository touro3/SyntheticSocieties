"""Policy Intervention Sweep Runner — Phase 28.8.

Runs trust-building intervention experiments at four intensities
(0%, 5%, 10%, 20%) across multiple seeds and saves results to
analysis/tables/policy_intervention.json, then calls the plot script.

No GPU required — uses synthetic agent profiles and rule-based decisions.

Usage
-----
    python scripts/run_policy_intervention.py
    python scripts/run_policy_intervention.py --n-agents 300 --n-rounds 30 --n-seeds 5
    python scripts/run_policy_intervention.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.policy_intervention import aggregate_by_intensity, run_intervention_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy intervention sweep — Phase 28.8")
    parser.add_argument(
        "--intervention-round", type=int, default=15, help="Round at which trust-boost takes effect (default: 15)"
    )
    parser.add_argument("--n-agents", type=int, default=200, help="Agents per run (default: 200)")
    parser.add_argument("--n-rounds", type=int, default=30, help="Total simulation rounds (default: 30)")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per intensity (default: 3)")
    parser.add_argument(
        "--intensities",
        type=str,
        default="0.0,0.05,0.10,0.20",
        help="Comma-separated trust-boost intensities (default: 0.0,0.05,0.10,0.20)",
    )
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation — re-plot from existing JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path("analysis/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "policy_intervention.json"

    if not args.plot_only:
        intensities = tuple(float(x.strip()) for x in args.intensities.split(","))
        seeds = tuple(range(42, 42 + args.n_seeds))

        print(
            f"[policy_intervention] Running sweep: intensities={intensities} "
            f"× {len(seeds)} seeds, {args.n_agents} agents, {args.n_rounds} rounds"
        )

        results = run_intervention_sweep(
            intensities=intensities,
            intervention_round=args.intervention_round,
            n_agents=args.n_agents,
            n_rounds=args.n_rounds,
            seeds=seeds,
        )

        summaries = aggregate_by_intensity(results)

        print("\n[policy_intervention] Results:")
        print(f"  {'δ':>6}  {'pre':>6}  {'post':>6}  {'Δcoop':>8}  {'Gini':>6}  {'Wealth':>8}")
        print(f"  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 6}  {'-' * 8}")
        for s in summaries:
            print(
                f"  {s.intensity_pct:>6}  {s.coop_pre_mean:.3f}  {s.coop_post_mean:.3f}  "
                f"{s.delta_coop_mean:+.4f}  {s.gini_mean:.4f}  {s.wealth_mean_final:>8.1f}"
            )

        payload = {
            "intervention_round": args.intervention_round,
            "n_agents": args.n_agents,
            "n_rounds": args.n_rounds,
            "n_seeds": args.n_seeds,
            "summary": [
                {
                    "intensity": s.intensity,
                    "intensity_pct": s.intensity_pct,
                    "coop_pre": round(s.coop_pre_mean, 4),
                    "coop_post": round(s.coop_post_mean, 4),
                    "delta_coop": round(s.delta_coop_mean, 4),
                    "delta_coop_std": round(s.delta_coop_std, 4),
                    "gini_final": round(s.gini_mean, 4),
                    "wealth_mean_final": round(s.wealth_mean_final, 2),
                    "per_round_cooperation": [round(v, 4) for v in s.per_round_cooperation],
                }
                for s in summaries
            ],
        }

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[policy_intervention] Saved to {json_path}")

    if json_path.exists():
        print("[policy_intervention] Generating figures…")
        subprocess.run(
            [sys.executable, "scripts/plot_policy_intervention.py", "--input", str(json_path)],
            check=False,
        )
    else:
        print(f"[policy_intervention] JSON not found at {json_path}; skipping plot.")


if __name__ == "__main__":
    main()
