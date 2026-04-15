#!/usr/bin/env python3
"""Plot cross-model RLHF bias comparison (Phase 16).

Reads a results JSON produced by run_cross_model_comparison.py and generates:
  1. Grouped bar chart: RLHF bias index per model × condition (A vs B)
  2. Grouped bar chart: cooperation rate per model × condition
  3. Summary table printed to stdout

Output: analysis/figures/cross_model_bias_comparison.png

Usage:
    python scripts/plot_cross_model_comparison.py
    python scripts/plot_cross_model_comparison.py --results path/to/results.json
    python scripts/plot_cross_model_comparison.py --out path/to/output.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_RESULTS = Path("analysis/cross_model_results.json")
_DEFAULT_OUT = Path("analysis/figures/cross_model_bias_comparison.png")

_CONDITION_COLORS = {
    "A": "#d62728",  # red — ungrounded (high bias)
    "B": "#2ca02c",  # green — grounded (low bias)
}
_MODEL_LABELS = {
    "mistral-7b": "Mistral-7B",
    "llama3-8b": "Llama-3.1-8B",
    "gpt4o-mini": "GPT-4o-mini",
}


def _load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _group_by_model_condition(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Returns {model_id: {condition: row_dict}}."""
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        m = row["model_id"]
        c = row["condition"]
        grouped.setdefault(m, {})[c] = row
    return grouped


def plot_cross_model_comparison(
    results_path: Path = _DEFAULT_RESULTS,
    out_path: Path = _DEFAULT_OUT,
    dpi: int = 150,
) -> Path:
    """Generate and save the cross-model comparison figure.

    Args:
        results_path: JSON file from run_cross_model_comparison.py.
        out_path: Output PNG path.
        dpi: Figure DPI.

    Returns:
        Path to saved figure.
    """
    rows = _load_results(results_path)
    grouped = _group_by_model_condition(rows)
    models = sorted(grouped.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Cross-Model RLHF Bias Comparison\nCondition A = Ungrounded LLM  |  Condition B = BGF Grounded",
        fontsize=12,
        fontweight="bold",
    )

    x = np.arange(len(models))
    width = 0.35

    for ax, metric_key, ylabel, title in [
        (axes[0], "rlhf_bias_index", "RLHF Bias Index", "RLHF Bias Index (lower = less biased)"),
        (axes[1], "cooperation_rate", "Cooperation Rate", "Cooperation Rate"),
    ]:
        bars_a = []
        bars_b = []
        for m in models:
            conds = grouped[m]
            bars_a.append(conds.get("A", {}).get(metric_key, 0.0))
            bars_b.append(conds.get("B", {}).get(metric_key, 0.0))

        rects_a = ax.bar(
            x - width / 2, bars_a, width, label="Condition A (Ungrounded)", color=_CONDITION_COLORS["A"], alpha=0.85
        )
        rects_b = ax.bar(
            x + width / 2, bars_b, width, label="Condition B (Grounded)", color=_CONDITION_COLORS["B"], alpha=0.85
        )

        ax.set_xticks(x)
        ax.set_xticklabels([_MODEL_LABELS.get(m, m) for m in models], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        # Value labels
        for rect in list(rects_a) + list(rects_b):
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


def _print_table(rows: list[dict]) -> None:
    """Print a markdown-style comparison table to stdout."""
    from metrics.cross_model import CrossModelResult, build_comparison_table

    results = [
        CrossModelResult(
            model_id=r["model_id"],
            condition=r["condition"],
            cooperation_rate=r["cooperation_rate"],
            gini=r["gini"],
            rlhf_bias_index=r["rlhf_bias_index"],
            n_agents=r.get("n_agents", 0),
            n_rounds=r.get("n_rounds", 0),
        )
        for r in rows
    ]

    table = build_comparison_table(results)
    header = (
        f"{'Model':<20} {'Coop-A':>8} {'Bias-A':>8} {'Gini-A':>8}"
        f" {'Coop-B':>8} {'Bias-B':>8} {'Gini-B':>8} {'ΔBias%':>8} {'Effective':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in table:
        coop_a = f"{row.get('coop_rate_A', '?'):>8}"
        bias_a = f"{row.get('bias_A', '?'):>8}"
        gini_a = f"{row.get('gini_A', '?'):>8}"
        coop_b = f"{row.get('coop_rate_B', '?'):>8}"
        bias_b = f"{row.get('bias_B', '?'):>8}"
        gini_b = f"{row.get('gini_B', '?'):>8}"
        delta = f"{row.get('bias_reduction_pct', '?'):>8}"
        eff = f"{'Yes' if row.get('grounding_effective') else 'No':>10}"
        print(f"{row['model']:<20} {coop_a}{bias_a}{gini_a}{coop_b}{bias_b}{gini_b}{delta}{eff}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-model RLHF bias comparison")
    parser.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        print("Run scripts/run_cross_model_comparison.py first.")
        raise SystemExit(1)

    rows = _load_results(args.results)
    _print_table(rows)
    plot_cross_model_comparison(args.results, args.out, args.dpi)


if __name__ == "__main__":
    main()
