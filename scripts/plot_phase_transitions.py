#!/usr/bin/env python3
"""Plot phase transition analysis results.

Phase 18 — Emergent complexity analysis.

Generates a 4-panel figure:
  1. Cooperation rate vs bad-apple fraction (sigmoid fit overlay)
  2. Gini coefficient vs shock magnitude
  3. Cooperation rate + modularity vs network rewiring probability
  4. Wealth distribution log-log plot (power law fit)

Usage:
    python scripts/plot_phase_transitions.py
    python scripts/plot_phase_transitions.py --input analysis/tables/phase_transitions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.complexity import fit_power_law, sigmoid


def plot_sweep_panel(
    ax: plt.Axes,
    sweep_values: list[float],
    metric_values: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    fit_result: dict | None = None,
    color: str = "steelblue",
):
    """Plot a single sweep panel with optional sigmoid fit overlay."""
    x = np.array(sweep_values)
    y = np.array(metric_values)

    ax.scatter(x, y, color=color, s=30, zorder=3, alpha=0.8)
    ax.plot(x, y, color=color, alpha=0.4, linewidth=1)

    if fit_result and fit_result.get("is_transition"):
        x_fine = np.linspace(x.min(), x.max(), 200)
        params = fit_result["fit_params"]
        y_fit = sigmoid(x_fine, params["L"], params["k"], params["x0"], params["b"])
        ax.plot(
            x_fine,
            y_fit,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"Sigmoid fit (R²={fit_result['r_squared']:.2f})",
        )
        ax.axvline(
            fit_result["inflection_point"],
            color="red",
            linestyle=":",
            alpha=0.5,
            label=f"Inflection: {fit_result['inflection_point']:.2f}",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_power_law_panel(
    ax: plt.Axes,
    wealth_values: list[float],
    title: str = "Wealth Distribution (Log-Log)",
):
    """Plot wealth distribution on log-log scale with power law fit."""
    values = np.array(wealth_values)
    values = values[values > 0]

    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Log-log histogram
    log_bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 30)
    counts, edges = np.histogram(values, bins=log_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = counts > 0

    ax.scatter(centers[mask], counts[mask], color="steelblue", s=20, zorder=3)

    # Power law fit
    result = fit_power_law(values)
    if result["is_power_law"]:
        x_fit = np.linspace(result["xmin"], values.max(), 100)
        alpha = result["alpha"]
        # PDF of power law: p(x) = (alpha-1)/xmin * (x/xmin)^(-alpha)
        y_fit = ((alpha - 1) / result["xmin"]) * (x_fit / result["xmin"]) ** (-alpha)
        ax.plot(x_fit, y_fit, color="red", linewidth=2, linestyle="--", label=f"Power law (alpha={alpha:.2f})")
        ax.legend(fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wealth", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot phase transition results")
    parser.add_argument(
        "--input", default="analysis/tables/phase_transitions.json", help="Path to phase_transitions.json"
    )
    parser.add_argument("--output", default="analysis/figures/phase_transitions.png", help="Output figure path")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    # Load results (or use synthetic data for demo)
    if input_path.exists():
        with open(input_path) as f:
            results = json.load(f)
    else:
        print(f"WARNING: {input_path} not found. Generating demo plot with synthetic data.")
        results = _generate_demo_data()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase Transition Analysis", fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Bad apple sweep
    if "bad_apple" in results:
        data = results["bad_apple"]
        analysis = data.get("analysis", {}).get("cooperation_rate", {})
        plot_sweep_panel(
            axes[0, 0],
            data["sweep_values"],
            data["metrics"]["cooperation_rate"],
            xlabel="Bad Apple Fraction",
            ylabel="Cooperation Rate",
            title="Cooperation vs Adversarial Injection",
            fit_result=analysis,
        )

    # Panel 2: Shock magnitude sweep
    if "shock" in results:
        data = results["shock"]
        analysis = data.get("analysis", {}).get("gini", {})
        plot_sweep_panel(
            axes[0, 1],
            data["sweep_values"],
            data["metrics"]["gini"],
            xlabel="Shock Magnitude",
            ylabel="Gini Coefficient (Round 30)",
            title="Inequality vs Economic Shock",
            fit_result=analysis,
            color="darkorange",
        )

    # Panel 3: Network rewiring beta sweep
    if "beta" in results:
        data = results["beta"]
        coop_analysis = data.get("analysis", {}).get("cooperation_rate", {})
        plot_sweep_panel(
            axes[1, 0],
            data["sweep_values"],
            data["metrics"]["cooperation_rate"],
            xlabel="Rewiring Probability (beta)",
            ylabel="Cooperation Rate",
            title="Cooperation vs Network Topology",
            fit_result=coop_analysis,
            color="seagreen",
        )

    # Panel 4: Wealth power law (from last sweep point with data)
    wealth_data = []
    for sweep_name in ["bad_apple", "shock", "beta"]:
        if sweep_name in results:
            mw = results[sweep_name]["metrics"].get("mean_wealth", [])
            wealth_data.extend(mw)

    if wealth_data:
        plot_power_law_panel(axes[1, 1], wealth_data)
    else:
        axes[1, 1].text(0.5, 0.5, "No wealth data available", ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {output_path}")


def _generate_demo_data() -> dict:
    """Generate synthetic demo data for visualization when no real data exists."""
    rng = np.random.default_rng(42)

    bad_apple_x = [i * 0.02 for i in range(21)]
    bad_apple_coop = sigmoid(np.array(bad_apple_x), L=-0.45, k=18.0, x0=0.15, b=0.65) + rng.normal(0, 0.02, 21)

    shock_x = [i * 0.1 for i in range(11)]
    shock_gini = sigmoid(np.array(shock_x), L=0.35, k=6.0, x0=0.5, b=0.10) + rng.normal(0, 0.01, 11)

    beta_x = [i * 0.1 for i in range(11)]
    beta_coop = sigmoid(np.array(beta_x), L=-0.25, k=8.0, x0=0.4, b=0.55) + rng.normal(0, 0.015, 11)

    return {
        "bad_apple": {
            "sweep_values": bad_apple_x,
            "metrics": {
                "cooperation_rate": bad_apple_coop.tolist(),
                "gini": (0.3 + rng.normal(0, 0.02, 21)).tolist(),
                "mean_wealth": (50 + rng.normal(0, 5, 21)).tolist(),
            },
            "analysis": {},
        },
        "shock": {
            "sweep_values": shock_x,
            "metrics": {
                "cooperation_rate": (0.4 + rng.normal(0, 0.03, 11)).tolist(),
                "gini": shock_gini.tolist(),
                "mean_wealth": (80 - np.array(shock_x) * 30 + rng.normal(0, 3, 11)).tolist(),
            },
            "analysis": {},
        },
        "beta": {
            "sweep_values": beta_x,
            "metrics": {
                "cooperation_rate": beta_coop.tolist(),
                "gini": (0.25 + rng.normal(0, 0.02, 11)).tolist(),
                "mean_wealth": (60 + rng.normal(0, 4, 11)).tolist(),
            },
            "analysis": {},
        },
    }


if __name__ == "__main__":
    main()
