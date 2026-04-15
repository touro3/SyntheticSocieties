"""Plot ESS Feature Importance — Phase 28.3.

Generates two figures:
  1. Horizontal bar chart of logistic regression coefficients (Figure 11).
  2. Profile-depth ablation bar chart (Figure 12).

Usage
-----
    python scripts/plot_feature_importance.py
    python scripts/plot_feature_importance.py --input analysis/tables/feature_importance.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_FEATURE_LABELS = {
    "trust_people": "Interpersonal Trust",
    "trust_institutions": "Institutional Trust",
    "risk_tolerance": "Risk Tolerance",
    "social_activity": "Social Activity",
    "life_satisfaction": "Life Satisfaction",
    "happiness": "Happiness",
    "competitiveness": "Competitiveness",
    "leadership_preference": "Leadership Preference",
    "health_status": "Health Status",
    "religiosity": "Religiosity",
    "political_orientation": "Political Orientation",
    "immigration_attitude": "Immigration Attitude",
}


def plot_coefficients(coefficients: list[dict], output_path: Path) -> None:
    """Horizontal bar chart of logistic regression coefficients."""
    # Sort by coefficient value (positive → cooperative, negative → avoidant)
    sorted_coefs = sorted(coefficients, key=lambda c: c["coefficient"])

    features = [_FEATURE_LABELS.get(c["feature"], c["feature"]) for c in sorted_coefs]
    coef_vals = [c["coefficient"] for c in sorted_coefs]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in coef_vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(features, coef_vals, color=colors, edgecolor="white", height=0.65)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Logistic Regression Coefficient (z-scored features)", fontsize=11)
    ax.set_title(
        "ESS Feature Importance: Predictors of Cooperation\n(Condition D — Rule-Based ESS Baseline)",
        fontsize=12,
        pad=14,
    )

    pos_patch = mpatches.Patch(color="#2ecc71", label="Promotes cooperation")
    neg_patch = mpatches.Patch(color="#e74c3c", label="Reduces cooperation")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    # Annotate bars with coefficient values
    for bar, val in zip(bars, coef_vals):
        x = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=8.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_feature_importance] Saved coefficient plot → {output_path}")


def plot_ablation(ablation_table: dict[str, float], output_path: Path) -> None:
    """Bar chart: profile depth (minimal / medium / full) vs. train accuracy."""
    levels = list(ablation_table.keys())
    accuracies = [ablation_table[l] for l in levels]
    level_labels = {
        "minimal": "Minimal\n(trust, risk)",
        "medium": "Medium\n(+social, life-sat)",
        "full": "Full\n(all 12 ESS dims)",
    }
    labels = [level_labels.get(l, l) for l in levels]

    colors = ["#95a5a6", "#3498db", "#2c3e50"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, accuracies, color=colors[: len(levels)], edgecolor="white", width=0.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Logistic Regression Train Accuracy", fontsize=11)
    ax.set_title(
        "Profile Richness vs. Cooperation Prediction\n(Monotonic improvement with ESS dimension depth)",
        fontsize=11,
        pad=12,
    )

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_feature_importance] Saved ablation plot → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="analysis/tables/feature_importance.json",
        help="Path to feature_importance.json produced by run_feature_importance.py",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[plot_feature_importance] Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    figures_dir = Path("analysis/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_coefficients(
        data["coefficients"],
        figures_dir / "feature_importance_coefficients.png",
    )

    if data.get("ablation_table"):
        plot_ablation(
            data["ablation_table"],
            figures_dir / "feature_importance_ablation.png",
        )


if __name__ == "__main__":
    main()
