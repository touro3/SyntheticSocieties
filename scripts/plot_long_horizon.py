"""Long-Horizon Persona Drift Figure — Phase 28.5.

Two-panel publication figure:
  A) Fidelity vs round for grounded vs ungrounded (shaded 95% CI bands)
  B) Decay rate comparison (bar chart)

Usage
-----
    python scripts/plot_long_horizon.py
    python scripts/plot_long_horizon.py --input analysis/tables/long_horizon_persona_drift.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot long-horizon persona drift")
    parser.add_argument(
        "--input",
        type=str,
        default="analysis/tables/long_horizon_persona_drift.json",
    )
    parser.add_argument("--output-dir", type=str, default="analysis/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[plot_long_horizon] JSON not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    results = data["results"]
    n_rounds = data["n_rounds"]

    grounded = results.get("grounded", {})
    ungrounded = results.get("ungrounded", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: Fidelity over rounds ────────────────────────────────────
    ax = axes[0]

    for cond, color, label in [
        (grounded, "#1f6fb2", "Grounded (Condition B)"),
        (ungrounded, "#e74c3c", "Ungrounded (Condition A)"),
    ]:
        if not cond:
            continue
        rounds = cond["rounds"]
        means = cond["fidelity_mean"]
        lo = cond["fidelity_ci_lower"]
        hi = cond["fidelity_ci_upper"]

        ax.plot(rounds, means, color=color, lw=2.5, label=label)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.15)

    ax.axhline(0.5, color="gray", linestyle="--", lw=1.2, label="Fidelity threshold (0.5)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Persona Fidelity")
    ax.set_title(f"A  Persona Fidelity over T={n_rounds} Rounds")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel B: Decay rate comparison ──────────────────────────────────
    ax = axes[1]

    conds = ["grounded", "ungrounded"]
    decay_means = [results.get(c, {}).get("decay_rate_mean", 0.0) for c in conds]
    decay_stds = [results.get(c, {}).get("decay_rate_std", 0.0) for c in conds]
    colors = ["#1f6fb2", "#e74c3c"]
    x = np.arange(2)

    bars = ax.bar(
        x, decay_means, yerr=decay_stds, capsize=6,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Grounded\n(Condition B)", "Ungrounded\n(Condition A)"])
    ax.set_ylabel("Persona Decay Rate (slope per round)")
    ax.set_title("B  Fidelity Decay Rate Comparison")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, decay_means, decay_stds):
        ypos = bar.get_height() + (std if mean >= 0 else -std) + 0.00005
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            ypos,
            f"{mean:+.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Annotation: which condition is worse
    ax.annotate(
        "More negative = faster decay\n(character capture failure)",
        xy=(0.5, 0.05),
        xycoords="axes fraction",
        ha="center",
        fontsize=8.5,
        color="gray",
    )

    fig.suptitle(
        f"Long-Horizon Analysis: Persona Stability over T={n_rounds} Rounds\n"
        f"({data['n_agents']} agents, {data['n_seeds']} seeds, window={data['window']})",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    out_path = out_dir / "persona_drift_long_horizon.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_long_horizon] Saved → {out_path}")


if __name__ == "__main__":
    main()
