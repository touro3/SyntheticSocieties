"""Policy Intervention Figure — Phase 28.8.

Three-panel figure:
  A) Cooperation rate over rounds for each intensity (lines + vertical bar at intervention)
  B) ΔCooperation (post − pre) by intensity (bar chart with ± 1σ error bars)
  C) Gini coefficient at final round by intensity

Usage
-----
    python scripts/plot_policy_intervention.py
    python scripts/plot_policy_intervention.py --input analysis/tables/policy_intervention.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot policy intervention results")
    parser.add_argument(
        "--input",
        type=str,
        default="analysis/tables/policy_intervention.json",
        help="Path to policy_intervention.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/figures",
        help="Output directory for figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[plot_policy_intervention] JSON not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    summary = data["summary"]
    intervention_round = data["intervention_round"]
    n_rounds = data["n_rounds"]

    intensities = [s["intensity"] for s in summary]
    labels = [s["intensity_pct"] for s in summary]
    delta_coops = [s["delta_coop"] for s in summary]
    delta_stds = [s["delta_coop_std"] for s in summary]
    ginis = [s["gini_final"] for s in summary]
    rounds = list(range(n_rounds))

    # Color palette — blue family, increasing saturation
    colors = ["#aec6e8", "#5ba3d9", "#1f6fb2", "#0a3d6b"]
    if len(colors) < len(summary):
        import matplotlib.cm as cm
        cmap = cm.get_cmap("Blues", len(summary) + 2)
        colors = [cmap(i + 1) for i in range(len(summary))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Panel A: Cooperation rate over rounds ────────────────────────────
    ax = axes[0]
    for s, color, label in zip(summary, colors, labels):
        per_round = s["per_round_cooperation"]
        ax.plot(rounds, per_round, color=color, lw=2.0, label=f"δ={label}")

    ax.axvline(
        x=intervention_round,
        color="#e74c3c",
        linestyle="--",
        lw=1.5,
        label=f"Intervention (r={intervention_round})",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("A  Cooperation Rate Over Time")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel B: ΔCooperation by intensity ──────────────────────────────
    ax = axes[1]
    x = np.arange(len(summary))
    bars = ax.bar(x, delta_coops, yerr=delta_stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Trust-boost Intensity δ")
    ax.set_ylabel("ΔCooperation (post − pre)")
    ax.set_title("B  Cooperation Gain by Intensity")
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val, err in zip(bars, delta_coops, delta_stds):
        ypos = bar.get_height() + err + 0.002
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            ypos,
            f"{val:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ── Panel C: Gini by intensity ───────────────────────────────────────
    ax = axes[2]
    ax.bar(x, ginis, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0.31, color="#e67e22", linestyle="--", lw=1.5,
               label="EU Median Gini ≈ 0.31")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Trust-boost Intensity δ")
    ax.set_ylabel("Gini Coefficient (final round)")
    ax.set_title("C  Inequality at Simulation End")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.set_ylim(0, max(max(ginis) * 1.3, 0.4))
    ax.grid(axis="y", alpha=0.3)

    for i, (val, label) in enumerate(zip(ginis, labels)):
        ax.text(i, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Policy Intervention Analysis: Trust-Building at Round 15\n"
        "(Rule-based ESS policy, 200 agents, 30 rounds, 3 seeds)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    out_path = out_dir / "policy_intervention_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_policy_intervention] Saved → {out_path}")


if __name__ == "__main__":
    main()
