"""Wrong-Culture Negative Control Figure.

Two-panel figure:
  A) BRM composite by condition — bar chart with ± std error bars
  B) Actual vs. empirical cooperation rate — shows the source of BRM gap

Usage
-----
    python scripts/plot_negative_control.py
    python scripts/plot_negative_control.py --input analysis/tables/negative_control.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="analysis/tables/negative_control.json")
    parser.add_argument("--output-dir", default="analysis/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[plot_negative_control] Not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})

    summary = data["summary"]
    conds = ["matched", "mismatched", "ungrounded"]
    labels = ["Matched\n(Nordic → Nordic)", "Mismatched\n(Nordic → Eastern)", "Ungrounded\n(Flat → Eastern)"]
    colors = ["#1f6fb2", "#e67e22", "#e74c3c"]

    brm_means = [summary[c]["brm_mean"] for c in conds]
    brm_stds = [summary[c]["brm_std"] for c in conds]
    act_coops = [summary[c]["actual_coop_mean"] for c in conds]
    emp_coops = [summary[c]["emp_coop_rate"] for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(3)

    # ── Panel A: BRM composite ──────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, brm_means, yerr=brm_stds, capsize=6, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("BRM Composite Score")
    ax.set_title("A  BRM by Grounding Condition")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, brm_means, brm_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Directionality annotation
    directionality = data.get("directionality_pass", False)
    ax.annotate(
        f"Directionality: {'PASS ✓' if directionality else 'FAIL ✗'}\nMatched > Mismatched > Ungrounded",
        xy=(0.5, 0.06),
        xycoords="axes fraction",
        ha="center",
        fontsize=8.5,
        color="#155724" if directionality else "#721c24",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#d4edda" if directionality else "#f8d7da",
            edgecolor="#c3e6cb" if directionality else "#f5c6cb",
            alpha=0.8,
        ),
    )

    # ── Panel B: Actual vs. Empirical cooperation ───────────────────────
    ax = axes[1]
    width = 0.35
    ax.bar(x - width / 2, act_coops, width, label="Actual (simulated)", color=colors, edgecolor="black", linewidth=0.7)
    ax.bar(
        x + width / 2,
        emp_coops,
        width,
        label="Empirical benchmark",
        color=[c + "80" for c in ["#1f6fb2", "#e67e22", "#e74c3c"]],
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
    )

    # Fix the semi-transparent color strings — use actual alpha parameter
    ax.cla()
    bar_act = ax.bar(
        x - width / 2, act_coops, width, label="Actual (simulated)", color=colors, edgecolor="black", linewidth=0.7
    )
    bar_emp = ax.bar(
        x + width / 2,
        emp_coops,
        width,
        label="Empirical benchmark",
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.4,
        hatch="//",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("B  Actual vs. Benchmark Cooperation Rate")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Gap arrows
    for i, (act, emp) in enumerate(zip(act_coops, emp_coops)):
        gap = abs(act - emp)
        if gap > 0.01:
            ymax = max(act, emp) + 0.03
            ax.annotate(f"Δ={gap:.2f}", xy=(i, ymax), ha="center", fontsize=8, color="#666666")

    fig.suptitle(
        "Negative Control: Wrong-Culture Grounding\n"
        "Nordic profiles grounded against Nordic (matched) vs. Eastern (mismatched) vs. Ungrounded",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    out_path = out_dir / "negative_control_brm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_negative_control] Saved → {out_path}")


if __name__ == "__main__":
    main()
