#!/usr/bin/env python3
"""Run parameter sweeps for phase transition analysis.

Phase 18 — Emergent complexity analysis.

This script orchestrates simulation runs across parameter sweeps,
collects results, and applies phase transition detection from
metrics/complexity.py.

Sweeps:
  1. Bad apple fraction:  0% to 40% in 2% steps
  2. Shock magnitude:     0% to 100% in 10% steps
  3. Network rewiring:    beta 0.0 to 1.0 in 0.1 steps

Usage:
    python scripts/run_phase_transition_sweeps.py [--no-llm]
    python scripts/run_phase_transition_sweeps.py --analyze-only

The --analyze-only flag skips simulation runs and analyzes existing
results from experiments/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.complexity import analyze_sweep_results, fit_power_law


def run_sweep(
    sweep_name: str,
    param_values: list[float],
    config_overrides: list[list[str]],
    no_llm: bool = False,
) -> list[str]:
    """Run simulation for each parameter value and return experiment IDs.

    Args:
        sweep_name: Name of the sweep (e.g., 'bad_apple').
        param_values: Parameter values to sweep.
        config_overrides: Hydra overrides for each parameter value.
        no_llm: If True, skip LLM-dependent runs.

    Returns:
        List of experiment directory names.
    """
    exp_ids = []

    for i, (val, overrides) in enumerate(zip(param_values, config_overrides)):
        exp_id = f"sweep_{sweep_name}_{i:03d}_val{val:.2f}"
        cmd = [
            sys.executable, "scripts/run_full_pipeline.py",
            "--experiment-id", exp_id,
        ] + overrides

        if no_llm:
            cmd.append("--no-llm")

        print(f"[{sweep_name}] Running {exp_id} (value={val:.3f})...")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           cwd=str(PROJECT_ROOT), timeout=3600)
            exp_ids.append(exp_id)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  WARNING: {exp_id} failed: {e}")

    return exp_ids


def collect_sweep_metrics(
    exp_ids: list[str],
    experiments_root: Path,
) -> dict[str, list[float]]:
    """Collect metrics from completed sweep experiments.

    Returns:
        {metric_name: [value_per_sweep_point]}
    """
    cooperation_rates = []
    gini_values = []
    mean_wealths = []

    for exp_id in exp_ids:
        summary_path = experiments_root / exp_id / "summary.json"
        if not summary_path.exists():
            cooperation_rates.append(float("nan"))
            gini_values.append(float("nan"))
            mean_wealths.append(float("nan"))
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        behavior = summary.get("behavior", summary.get("event_behavior", {}))
        coop = behavior.get("cooperation_rate", 0.0)
        gini = summary.get("wealth", {}).get("gini", 0.0)
        mean_w = summary.get("wealth", {}).get("mean", 0.0)

        cooperation_rates.append(coop)
        gini_values.append(gini)
        mean_wealths.append(mean_w)

    return {
        "cooperation_rate": cooperation_rates,
        "gini": gini_values,
        "mean_wealth": mean_wealths,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase transition sweep analysis")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM-dependent runs")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip runs, analyze existing results")
    args = parser.parse_args()

    experiments_root = PROJECT_ROOT / "experiments"
    output_dir = PROJECT_ROOT / "analysis" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Define sweeps ────────────────────────────────────────────────────

    # Sweep 1: Bad apple fraction (0% to 40%)
    bad_apple_values = [i * 0.02 for i in range(21)]
    bad_apple_overrides = [
        [f"simulation.bad_apple_fraction={v}"]
        for v in bad_apple_values
    ]

    # Sweep 2: Shock magnitude (0% to 100%)
    shock_values = [i * 0.1 for i in range(11)]
    shock_overrides = [
        [f"simulation.shock_magnitude={v}", "simulation.shock_round=15"]
        for v in shock_values
    ]

    # Sweep 3: Network rewiring beta (0.0 to 1.0)
    beta_values = [i * 0.1 for i in range(11)]
    beta_overrides = [
        [f"network.rewiring_probability={v}", "network.type=small_world"]
        for v in beta_values
    ]

    sweeps = {
        "bad_apple": (bad_apple_values, bad_apple_overrides),
        "shock": (shock_values, shock_overrides),
        "beta": (beta_values, beta_overrides),
    }

    all_results = {}

    for sweep_name, (values, overrides) in sweeps.items():
        if not args.analyze_only:
            exp_ids = run_sweep(sweep_name, values, overrides, no_llm=args.no_llm)
        else:
            # Reconstruct expected exp_ids
            exp_ids = [
                f"sweep_{sweep_name}_{i:03d}_val{v:.2f}"
                for i, v in enumerate(values)
            ]

        metrics = collect_sweep_metrics(exp_ids, experiments_root)

        # Filter NaN values
        valid_mask = ~np.isnan(metrics["cooperation_rate"])
        if valid_mask.sum() < 5:
            print(f"[{sweep_name}] Not enough valid results ({valid_mask.sum()}/min 5)")
            continue

        valid_x = np.array(values)[valid_mask]
        valid_metrics = {
            k: np.array(v)[valid_mask] for k, v in metrics.items()
        }

        # Analyze phase transitions
        analysis = analyze_sweep_results(valid_x, valid_metrics)
        all_results[sweep_name] = {
            "sweep_values": valid_x.tolist(),
            "metrics": {k: v.tolist() for k, v in valid_metrics.items()},
            "analysis": {
                k: {
                    kk: (vv if not isinstance(vv, float) or not np.isnan(vv) else None)
                    for kk, vv in v.items()
                }
                for k, v in analysis.items()
            },
        }

        for metric_name, result in analysis.items():
            status = "TRANSITION" if result["is_transition"] else "no transition"
            print(
                f"[{sweep_name}/{metric_name}] {status} "
                f"(inflection={result['inflection_point']:.3f}, "
                f"R²={result['r_squared']:.3f})"
            )

    # Save results
    output_path = output_dir / "phase_transitions.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
