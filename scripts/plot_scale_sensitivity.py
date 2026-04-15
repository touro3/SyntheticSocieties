"""Scale Sensitivity Figure.

Two-panel figure:
  A) BRM vs. N for grounded/ungrounded with 95% CI bands
  B) Grounding effect (BRM gap) vs. N — shows where grounding becomes critical

Usage
-----
    python scripts/plot_scale_sensitivity.py
    python scripts/plot_scale_sensitivity.py --input analysis/tables/scale_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="analysis/tables/scale_sensitivity.json")
    parser.add_argument("--output-dir", default="analysis/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"[plot_scale_sensitivity] Not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})

    summaries = data["summaries"]

    # Separate conditions
    grounded = {s["n_agents"]: s for s in summaries if s["condition"] == "grounded"}
    ungrounded = {s["n_agents"]: s for s in summaries if s["condition"] == "ungrounded"}
    ns = sorted(grounded.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: BRM vs. N ──────────────────────────────────────────────
    ax = axes[0]

    for cond_dict, color, label in [
        (grounded, "#1f6fb2", "Grounded (Condition B)"),
        (ungrounded, "#e74c3c", "Ungrounded (Condition A)"),
    ]:
        means = [cond_dict[n]["brm_mean"] for n in ns]
        lowers = [cond_dict[n]["brm_ci_lower"] for n in ns]
        uppers = [cond_dict[n]["brm_ci_upper"] for n in ns]

        ax.plot(ns, means, color=color, lw=2.5, marker="o", markersize=5, label=label)
        ax.fill_between(ns, lowers, uppers, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Population Size N (log scale)")
    ax.set_ylabel("BRM Composite")
    ax.set_title("A  Behavioral Realism vs. Population Size")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    # ── Panel B: Grounding effect vs. N ────────────────────────────────
    ax = axes[1]

    effects = [grounded[n]["brm_mean"] - ungrounded[n]["brm_mean"] for n in ns]
    effect_lo = [grounded[n]["brm_ci_lower"] - ungrounded[n]["brm_ci_upper"] for n in ns]
    effect_hi = [grounded[n]["brm_ci_upper"] - ungrounded[n]["brm_ci_lower"] for n in ns]

    ax.plot(ns, effects, color="#2ecc71", lw=2.5, marker="s", markersize=5, label="Grounding effect (ΔBRM)")
    ax.fill_between(ns, effect_lo, effect_hi, color="#2ecc71", alpha=0.2)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Population Size N (log scale)")
    ax.set_ylabel("ΔBRM (Grounded − Ungrounded)")
    ax.set_title("B  Grounding Effect vs. Population Size")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])

    for n, eff in zip(ns, effects):
        ax.text(n, eff + 0.005, f"{eff:+.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Scale Sensitivity Analysis ({data['n_seeds']} seeds per point, "
        f"{data['n_rounds']} rounds, rule-based ESS proxy)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    out_path = out_dir / "scale_sensitivity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_scale_sensitivity] Saved → {out_path}")


if __name__ == "__main__":
    main()
