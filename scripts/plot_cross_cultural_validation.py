#!/usr/bin/env python3
"""Plot cross-cultural validation scatter.

Reads analysis/cross_cultural_results.json and produces:
    analysis/figures/cross_cultural_validation.png

Scatter plot: ESS mean trust (x) vs simulated cooperation rate (y),
one point per cluster, with labels, and Pearson r / Spearman rho annotated
in the figure corner.

Usage:
    python scripts/plot_cross_cultural_validation.py
    python scripts/plot_cross_cultural_validation.py --input analysis/cross_cultural_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_RESULTS_PATH = PROJECT_ROOT / "analysis" / "cross_cultural_results.json"
_FIGURE_PATH = PROJECT_ROOT / "analysis" / "figures" / "cross_cultural_validation.png"

# Cluster display labels
_CLUSTER_LABELS = {
    "nordic": "Nordic\n(NO/SE/FI/DK)",
    "southern": "Southern\n(ES/IT/PT/GR)",
    "eastern": "Eastern\n(PL/CZ/HU/SK)",
}

# Cluster marker colors (colorblind-safe)
_CLUSTER_COLORS = {
    "nordic": "#2196F3",  # blue
    "southern": "#FF9800",  # orange
    "eastern": "#4CAF50",  # green
}


def plot_cross_cultural_validation(
    results_path: Path = _RESULTS_PATH,
    figure_path: Path = _FIGURE_PATH,
) -> None:
    """Load results and produce the scatter plot.

    Args:
        results_path: Path to cross_cultural_results.json.
        figure_path: Output PNG path.

    Raises:
        FileNotFoundError: If results_path does not exist.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}\nRun scripts/run_cross_cultural.py first.")

    data = json.loads(results_path.read_text())
    cluster_results = data["cluster_results"]
    correlation = data.get("correlation", {})

    pearson_r = correlation.get("pearson_r", float("nan"))
    spearman_rho = correlation.get("spearman_rho", float("nan"))
    spearman_p = correlation.get("spearman_p", float("nan"))
    gradient_recovered = correlation.get("gradient_recovered", False)

    fig, ax = plt.subplots(figsize=(6, 6))

    x_vals = []
    y_vals = []

    for r in cluster_results:
        name = r["cluster_name"]
        x = r["ess_mean_trust"]
        y = r["simulated_cooperation_rate"]
        x_vals.append(x)
        y_vals.append(y)

        color = _CLUSTER_COLORS.get(name, "#888888")
        label = _CLUSTER_LABELS.get(name, name)

        ax.scatter(x, y, s=120, color=color, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

    # Fit line through the 3 points
    if len(x_vals) >= 2:
        import numpy as np

        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = [min(x_vals) - 0.02, max(x_vals) + 0.02]
        ax.plot(
            x_line,
            [p(xi) for xi in x_line],
            color="#666666",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            alpha=0.7,
        )

    # Correlation annotation (upper left corner)
    annotation = f"Pearson r = {pearson_r:+.3f}\nSpearman \u03c1 = {spearman_rho:+.3f}  (p = {spearman_p:.3f})"
    if gradient_recovered:
        annotation += "\nGradient recovered"

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    # Axes and labels
    ax.set_xlabel("ESS-11 mean interpersonal trust (0-1)", fontsize=11)
    ax.set_ylabel("Simulated cooperation rate", fontsize=11)
    ax.set_title("Cross-Cultural ESS Validation\nESS Trust vs Simulated Cooperation", fontsize=11)

    # Publication style: only light horizontal grid, no vertical grid or box frame
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Margins
    ax.margins(0.20)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-cultural validation scatter")
    parser.add_argument(
        "--input",
        type=Path,
        default=_RESULTS_PATH,
        help="Path to cross_cultural_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_FIGURE_PATH,
        help="Output PNG path",
    )
    args = parser.parse_args()
    plot_cross_cultural_validation(results_path=args.input, figure_path=args.output)


if __name__ == "__main__":
    main()
