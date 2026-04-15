"""BRM Stability and Confidence Interval Figure.

Computes BRM composite from per-seed experiment data and plots:
  A) BRM per seed for Conditions A/B — violin + jitter overlay
  B) Bootstrap 95% CI on BRM(B) − BRM(A) difference

Reads from analysis/tables/grounding_comparison_seed_metrics.csv.
BRM is computed from: cooperation_rate, wealth_gini (from CSV) +
empirical ESS benchmarks.

Purpose: Validates that BRM(B) > BRM(A) is not a statistical artifact
and that the metric has low within-condition variance.

Output:
    analysis/figures/brm_stability.png

Usage
-----
    python scripts/plot_brm_stability.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.behavioral_realism import compute_composite_brm, compute_rlhf_bias_index
from metrics.statistical_inference import bootstrap_ci

# ESS empirical benchmarks
_EMP_GINI = 0.31
_EMP_COOP_A = 0.35  # ungrounded expected (human rate)
_EMP_COOP_B = 0.35  # same target (grounded should match; A fails to)
_EMP_WEALTH_MEAN = 90.0


def _approx_wealth_dist(wealth_mean: float, gini: float, n: int = 100, rng_seed: int = 42) -> list[float]:
    """Approximate lognormal wealth distribution matching given mean and Gini."""
    # For lognormal: Gini ≈ 2*Φ(σ/√2) - 1, so σ ≈ √2 * Φ⁻¹((Gini+1)/2)
    # Simplified: use sigma=0.5 as reasonable default
    sigma = max(0.15, min(1.2, gini * 2.0))
    mu = np.log(max(wealth_mean, 1.0)) - 0.5 * sigma**2
    rng = np.random.default_rng(rng_seed)
    return list(np.clip(rng.lognormal(mu, sigma, n), 1.0, wealth_mean * 5))


def load_and_compute_brm(csv_path: Path) -> list[dict]:
    """Load seed metrics CSV and compute BRM for each row."""
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition_key = row["condition_key"]
            seed = int(row["seed"])
            coop = float(row["cooperation_rate"])
            gini = float(row["wealth_gini"])
            wealth_mean = float(row["wealth_mean"])
            work_rate = float(row["work_rate"])
            save_rate = float(row["save_rate"])

            # Only process primary LLM conditions (A and B)
            if condition_key not in ("pure_llm_ess_persona", "grounded_llm_ess_persona"):
                continue

            condition = "A" if condition_key == "pure_llm_ess_persona" else "B"

            sim_wealth = _approx_wealth_dist(wealth_mean, gini, n=100, rng_seed=seed)
            emp_wealth = _approx_wealth_dist(_EMP_WEALTH_MEAN, _EMP_GINI, n=100, rng_seed=99)

            brm = compute_composite_brm(
                sim_wealth=sim_wealth,
                emp_wealth=emp_wealth,
                sim_gini=gini,
                emp_gini=_EMP_GINI,
                sim_coop_rate=coop,
                emp_coop_rate=_EMP_COOP_B,
                temporal_stability_jsd=0.05,
            )

            act_dist = {"work": work_rate, "save": save_rate, "cooperate": coop}
            b_rlhf = compute_rlhf_bias_index(act_dist)

            records.append(
                {
                    "condition": condition,
                    "condition_key": condition_key,
                    "seed": seed,
                    "brm_composite": round(brm["composite"], 4),
                    "brm_coop": round(brm["coop_component"], 4),
                    "brm_gini": round(brm["gini_component"], 4),
                    "b_rlhf": round(b_rlhf, 4),
                    "coop_rate": coop,
                    "gini": gini,
                }
            )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BRM stability and CIs")
    parser.add_argument("--csv", default="analysis/tables/grounding_comparison_seed_metrics.csv")
    parser.add_argument("--output-dir", default="analysis/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[plot_brm_stability] CSV not found: {csv_path}")
        return

    records = load_and_compute_brm(csv_path)
    if not records:
        print("[plot_brm_stability] No records found for Conditions A/B.")
        return

    brm_a = [r["brm_composite"] for r in records if r["condition"] == "A"]
    brm_b = [r["brm_composite"] for r in records if r["condition"] == "B"]

    if not brm_a or not brm_b:
        print(f"[plot_brm_stability] Insufficient data: A={len(brm_a)}, B={len(brm_b)}")
        return

    _, ci_a_lo, ci_a_hi = bootstrap_ci(brm_a, n_bootstrap=2000, random_state=42)
    _, ci_b_lo, ci_b_hi = bootstrap_ci(brm_b, n_bootstrap=2000, random_state=42)

    # Difference distribution
    differences = [b - a for b, a in zip(brm_b, brm_a)]
    _, ci_diff_lo, ci_diff_hi = bootstrap_ci(differences, n_bootstrap=2000, random_state=42)

    # Pack into simple objects for readability below
    class CI:
        def __init__(self, lo, hi):
            self.lower = lo
            self.upper = hi

    ci_a = CI(ci_a_lo, ci_a_hi)
    ci_b = CI(ci_b_lo, ci_b_hi)
    ci_diff = CI(ci_diff_lo, ci_diff_hi)

    print(f"[plot_brm_stability] BRM(A): {np.mean(brm_a):.4f} [{ci_a.lower:.4f}, {ci_a.upper:.4f}]")
    print(f"[plot_brm_stability] BRM(B): {np.mean(brm_b):.4f} [{ci_b.lower:.4f}, {ci_b.upper:.4f}]")
    print(f"[plot_brm_stability] ΔBRM:   {np.mean(differences):+.4f} [{ci_diff.lower:+.4f}, {ci_diff.upper:+.4f}]")
    print(f"[plot_brm_stability] Non-overlapping CIs: {'YES ✓' if ci_a.upper < ci_b.lower else 'NO — check scale'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: BRM per seed — violin + strip ──────────────────────────
    ax = axes[0]

    data_by_cond = {"A": brm_a, "B": brm_b}
    positions = [1, 2]
    colors = {"A": "#e74c3c", "B": "#1f6fb2"}
    labels_map = {"A": "Condition A\n(Ungrounded)", "B": "Condition B\n(Grounded)"}

    for pos, (cond, values) in zip(positions, data_by_cond.items()):
        vp = ax.violinplot(values, positions=[pos], showmedians=True, showextrema=True)
        for part in ["bodies", "cmedians", "cmins", "cmaxes", "cbars"]:
            if part in vp:
                items = vp[part] if part == "bodies" else [vp[part]]
                for item in items:
                    item.set_color(colors[cond])
                    item.set_alpha(0.7 if part == "bodies" else 1.0)

        # Jitter overlay
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(values))
        ax.scatter([pos + j for j in jitter], values, color=colors[cond], alpha=0.9, s=40, zorder=3)

        # CI bar
        ci = ci_a if cond == "A" else ci_b
        ax.plot([pos - 0.15, pos + 0.15], [ci.lower, ci.lower], color=colors[cond], lw=2, linestyle="--")
        ax.plot([pos - 0.15, pos + 0.15], [ci.upper, ci.upper], color=colors[cond], lw=2, linestyle="--")

    ax.set_xticks(positions)
    ax.set_xticklabels([labels_map[c] for c in data_by_cond])
    ax.set_ylabel("BRM Composite Score")
    ax.set_title("A  BRM Distribution per Seed")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate means with CIs
    for pos, (cond, values) in zip(positions, data_by_cond.items()):
        ci = ci_a if cond == "A" else ci_b
        ax.text(
            pos,
            max(values) + 0.04,
            f"μ={np.mean(values):.3f}\n[{ci.lower:.3f}, {ci.upper:.3f}]",
            ha="center",
            fontsize=8.5,
            color=colors[cond],
        )

    # ── Panel B: Bootstrap distribution of ΔBRM ────────────────────────
    ax = axes[1]

    # Bootstrap 2000 resamples of ΔBRM
    rng = np.random.default_rng(42)
    n_boot = 2000
    boot_diffs = []
    for _ in range(n_boot):
        sample_b = rng.choice(brm_b, size=len(brm_b), replace=True)
        sample_a = rng.choice(brm_a, size=len(brm_a), replace=True)
        boot_diffs.append(float(np.mean(sample_b) - np.mean(sample_a)))

    ax.hist(boot_diffs, bins=40, color="#2ecc71", edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axvline(0, color="black", lw=1.5, linestyle="--", label="No effect (0)")
    ax.axvline(ci_diff.lower, color="#27ae60", lw=1.5, linestyle=":", label="95% CI bounds")
    ax.axvline(ci_diff.upper, color="#27ae60", lw=1.5, linestyle=":")
    ax.axvline(np.mean(differences), color="#1a5e2a", lw=2.0, label=f"Observed Δ = {np.mean(differences):+.3f}")

    ax.set_xlabel("ΔBRM (B − A)")
    ax.set_ylabel("Bootstrap frequency")
    ax.set_title("B  Bootstrap Distribution of ΔBRM")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    overlap = ci_a.upper >= ci_b.lower
    ax.text(
        0.05,
        0.93,
        f"95% CI: [{ci_diff.lower:+.3f}, {ci_diff.upper:+.3f}]\n"
        f"{'CI excludes 0 → significant ✓' if ci_diff.lower > 0 else 'CI includes 0 → n.s.'}",
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#e8f5e9" if ci_diff.lower > 0 else "#ffeeba", alpha=0.8),
    )

    fig.suptitle(
        f"BRM Metric Stability: Conditions A vs. B\n({len(brm_a)} seeds per condition; bootstrap n=2,000)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    out_path = out_dir / "brm_stability.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_brm_stability] Saved → {out_path}")


if __name__ == "__main__":
    main()
