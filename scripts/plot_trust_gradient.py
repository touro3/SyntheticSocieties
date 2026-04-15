#!/usr/bin/env python3
"""Plot trust-gradient sub-population validation results.

Phase 17 — Trust-Gradient Sub-Population Validation.

Generates a 2-panel figure:
  1. Bar chart: ESS trust mean vs simulated cooperation rate by group
  2. Scatter: ESS trust vs simulated coop rate with Spearman r annotation

Usage:
    python scripts/plot_trust_gradient.py
    python scripts/plot_trust_gradient.py --input analysis/tables/trust_gradient.json
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

from metrics.trust_gradient import TRUST_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trust-gradient results")
    parser.add_argument("--input", default="analysis/tables/trust_gradient.json")
    parser.add_argument("--output", default="analysis/figures/trust_gradient.png")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    if not input_path.exists():
        print(f"WARNING: {input_path} not found. Run scripts/run_trust_gradient.py first.")
        sys.exit(0)

    with open(input_path) as f:
        data = json.load(f)

    group_results = data["group_results"]
    correlation = data["correlation"]

    group_names = [g.name for g in TRUST_GROUPS]
    ess_trust = [g.ess_reference_mean for g in TRUST_GROUPS]
    coop_rates = [group_results[n]["coop_rate"] for n in group_names]
    coop_stds = [group_results[n].get("coop_rate_std", 0.0) for n in group_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trust-Gradient Sub-Population Validation", fontsize=14, fontweight="bold")

    # ── Panel 1: Grouped bar chart ────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(group_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ess_trust, width, label="ESS Trust Mean", color="steelblue", alpha=0.85)
    bars2 = ax.bar(
        x + width / 2,
        coop_rates,
        width,
        label="Simulated Coop Rate",
        color="darkorange",
        alpha=0.85,
        yerr=coop_stds,
        capsize=4,
    )

    ax.set_xlabel("Trust Sub-Population", fontsize=10)
    ax.set_ylabel("Value [0, 1]", fontsize=10)
    ax.set_title("ESS Trust vs Simulated Cooperation", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-Trust", "") for n in group_names], rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 2: Scatter with regression line ────────────────────────────────
    ax2 = axes[1]
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9"]
    for i, (name, xt, ct) in enumerate(zip(group_names, ess_trust, coop_rates)):
        ax2.scatter(xt, ct, color=colors[i], s=100, zorder=5, label=name)

    # OLS line
    x_arr = np.array(ess_trust)
    y_arr = np.array(coop_rates)
    if np.std(x_arr) > 0 and np.std(y_arr) > 0:
        m, b = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min() - 0.05, x_arr.max() + 0.05, 100)
        ax2.plot(x_line, m * x_line + b, "k--", alpha=0.5, linewidth=1.5, label="OLS fit")

    r = correlation["spearman_r"]
    p = correlation["p_value"]
    sig_marker = "*" if correlation["is_significant"] else ""
    ax2.annotate(
        f"Spearman r = {r:.3f}{sig_marker}\np = {p:.3f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    ax2.set_xlabel("ESS Trust Mean (reference)", fontsize=10)
    ax2.set_ylabel("Simulated Cooperation Rate", fontsize=10)
    ax2.set_title("Gradient Recovery Scatter", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {output_path}")


if __name__ == "__main__":
    main()
