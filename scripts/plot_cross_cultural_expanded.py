"""Plot expanded cross-cultural validation results (6 clusters, multi-seed CIs).

Reads analysis/cross_cultural_expanded_results.json produced by
scripts/run_cross_cultural_expanded.py and generates:

  1. Primary scatter: ESS-11 mean trust (x) vs simulated cooperation rate (y)
     - 95% CI error bars per cluster
     - OLS regression line + 95% confidence band (bootstrap)
     - Pearson r and Spearman ρ annotation
     - Cluster labels with country codes

  2. Optional WVS inset / secondary x-axis when wvs_trust_pct is present.

Output: analysis/figures/cross_cultural_expanded.png (300 DPI)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_PATH = Path("analysis/cross_cultural_expanded_results.json")
OUTPUT_PATH = Path("analysis/figures/cross_cultural_expanded.png")

# Cluster display order (low → high trust)
CLUSTER_ORDER = ["eastern", "southern", "western", "anglo", "northern", "nordic"]

CLUSTER_COLORS = {
    "eastern":  "#d62728",   # red
    "southern": "#ff7f0e",   # orange
    "western":  "#2ca02c",   # green
    "anglo":    "#1f77b4",   # blue
    "northern": "#9467bd",   # purple
    "nordic":   "#17becf",   # cyan
}

CLUSTER_LABELS = {
    "eastern":  "Eastern\n(PL,CZ,HU,SK)",
    "southern": "Southern\n(IT,ES,PT,GR)",
    "western":  "Western\n(DE,BE,AT,FR)",
    "anglo":    "Anglo\n(GB,IE,EE)",
    "northern": "Northern\n(NL,CH,IS)",
    "nordic":   "Nordic\n(NO,SE,FI,DK)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Results file not found: {path}\nRun scripts/run_cross_cultural_expanded.py first.")
    with open(path) as f:
        return json.load(f)


def extract_cluster_data(results: dict) -> list[dict]:
    """Return list of per-cluster dicts with all fields needed for plotting."""
    clusters_raw = results.get("cluster_results", results.get("clusters", {}))
    rows = []
    for name in CLUSTER_ORDER:
        if name not in clusters_raw:
            continue
        c = clusters_raw[name]
        rows.append(
            {
                "name": name,
                "ess_mean_trust": c["ess_mean_trust"],
                "mean_coop": c["mean_cooperation_rate"],
                "ci_lower": c["ci_lower"],
                "ci_upper": c["ci_upper"],
                "std_coop": c.get("std_cooperation_rate", 0.0),
                "n_seeds": c.get("n_seeds", 1),
                "wvs_trust_pct": c.get("wvs_trust_pct"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def ols_regression(x: np.ndarray, y: np.ndarray):
    """Return slope, intercept, r_value, p_value, stderr."""
    return linregress(x, y)


def bootstrap_ci_band(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap 95% confidence band for OLS regression line."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    boot_preds = np.zeros((n_boot, len(x_grid)))
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) < 1e-12:
            boot_preds[i] = np.full(len(x_grid), np.mean(ys))
            continue
        slope, intercept, *_ = linregress(xs, ys)
        boot_preds[i] = slope * x_grid + intercept
    lo = np.percentile(boot_preds, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_plot(
    rows: list[dict],
    output_path: Path,
    show_wvs: bool = False,
) -> None:
    x = np.array([r["ess_mean_trust"] for r in rows])
    y = np.array([r["mean_coop"] for r in rows])
    yerr_lo = np.array([r["mean_coop"] - r["ci_lower"] for r in rows])
    yerr_hi = np.array([r["ci_upper"] - r["mean_coop"] for r in rows])

    # Correlations
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    # Regression
    slope, intercept, r_val, p_val, stderr = ols_regression(x, y)
    x_grid = np.linspace(x.min() - 0.02, x.max() + 0.02, 300)
    y_fit = slope * x_grid + intercept
    ci_lo, ci_hi = bootstrap_ci_band(x, y, x_grid)

    # ---------- figure ----------
    fig_w = 10.0
    fig_h = 7.0 if not show_wvs else 8.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Confidence band
    ax.fill_between(
        x_grid, ci_lo, ci_hi,
        color="#1f77b4", alpha=0.12, label="95% CI band (bootstrap)",
    )

    # Regression line
    ax.plot(
        x_grid, y_fit,
        color="#1f77b4", linewidth=1.6, linestyle="--",
        label=f"OLS fit (slope={slope:+.3f})",
    )

    # Data points with error bars
    for r in rows:
        name = r["name"]
        color = CLUSTER_COLORS.get(name, "gray")
        label_text = CLUSTER_LABELS.get(name, name)
        xi = r["ess_mean_trust"]
        yi = r["mean_coop"]
        ei_lo = yi - r["ci_lower"]
        ei_hi = r["ci_upper"] - yi

        ax.errorbar(
            xi, yi,
            yerr=[[ei_lo], [ei_hi]],
            fmt="o",
            color=color,
            markersize=9,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            zorder=5,
        )

        # Label offset: shift alternating clusters up/down to avoid overlap
        idx = CLUSTER_ORDER.index(name)
        y_offset = 0.022 if idx % 2 == 0 else -0.032
        ax.annotate(
            label_text,
            xy=(xi, yi),
            xytext=(xi + 0.004, yi + y_offset),
            fontsize=7.5,
            color=color,
            ha="left",
            va="center",
        )

    # Annotations
    stat_text = (
        f"Pearson r = {pr:+.3f}  (p = {pp:.3f})\n"
        f"Spearman ρ = {sr:+.3f}  (p = {sp:.3f})\n"
        f"N clusters = {len(rows)},  N seeds/cluster = {rows[0]['n_seeds']}"
    )
    ax.text(
        0.03, 0.97,
        stat_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )

    # Axes labels and title
    ax.set_xlabel("ESS-11 Country-Cluster Mean Interpersonal Trust", fontsize=12)
    ax.set_ylabel("Simulated Cooperation Rate (mean ± 95% CI)", fontsize=12)
    ax.set_title(
        "Cross-Cultural Validation: ESS Trust Gradient vs BGF Simulated Cooperation",
        fontsize=13, fontweight="bold", pad=12,
    )

    ax.set_xlim(x.min() - 0.06, x.max() + 0.06)
    ax.set_ylim(
        max(0.0, y.min() - 0.15),
        min(1.0, y.max() + 0.12),
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=CLUSTER_COLORS[n], label=n.capitalize())
        for n in CLUSTER_ORDER
        if any(r["name"] == n for r in rows)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        title="Cluster",
        title_fontsize=9,
        framealpha=0.85,
    )

    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=10)

    # ---------- WVS secondary x-axis ----------
    if show_wvs and all(r.get("wvs_trust_pct") is not None for r in rows):
        _add_wvs_inset(fig, ax, rows)

    try:
        fig.tight_layout()
    except Exception:
        pass  # inset axes may not be tight_layout-compatible; figure still renders correctly
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _add_wvs_inset(fig, ax_main, rows: list[dict]) -> None:
    """Add a small inset scatter using WVS trust % as secondary benchmark."""
    x_wvs = np.array([r["wvs_trust_pct"] for r in rows])
    y_coop = np.array([r["mean_coop"] for r in rows])

    # Inset axes: bottom-right corner of the figure
    ax_inset = fig.add_axes([0.62, 0.12, 0.28, 0.28])

    for r in rows:
        name = r["name"]
        color = CLUSTER_COLORS.get(name, "gray")
        ax_inset.errorbar(
            r["wvs_trust_pct"],
            r["mean_coop"],
            yerr=[[r["mean_coop"] - r["ci_lower"]], [r["ci_upper"] - r["mean_coop"]]],
            fmt="o", color=color, markersize=6, capsize=3, elinewidth=1.2,
        )

    if len(x_wvs) >= 2 and np.std(x_wvs) > 1e-6:
        sl, ic, *_ = linregress(x_wvs, y_coop)
        xg = np.linspace(x_wvs.min() - 2, x_wvs.max() + 2, 100)
        ax_inset.plot(xg, sl * xg + ic, "k--", linewidth=1.0)
        pr_wvs, _ = pearsonr(x_wvs, y_coop)
        ax_inset.set_title(f"WVS r={pr_wvs:+.2f}", fontsize=7, pad=3)

    ax_inset.set_xlabel("WVS Trust %", fontsize=7)
    ax_inset.set_ylabel("Sim. Coop", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.grid(True, alpha=0.25, linestyle=":")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot expanded cross-cultural validation results.")
    p.add_argument(
        "--results",
        default=str(RESULTS_PATH),
        help="Path to cross_cultural_expanded_results.json",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output PNG path (default: analysis/figures/cross_cultural_expanded.png)",
    )
    p.add_argument(
        "--wvs",
        action="store_true",
        help="Add WVS secondary validation inset",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.results))
    rows = extract_cluster_data(results)

    if not rows:
        sys.exit("No cluster data found in results file.")

    x = np.array([r["ess_mean_trust"] for r in rows])
    y = np.array([r["mean_coop"] for r in rows])
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    print(f"\nCross-Cultural Expanded Validation")
    print("=" * 50)
    print(f"{'Cluster':<12} {'ESS Trust':>10} {'Coop Mean':>10} {'CI Lower':>9} {'CI Upper':>9} {'N seeds':>8}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['name']:<12} "
            f"{r['ess_mean_trust']:>10.3f} "
            f"{r['mean_coop']:>10.3f} "
            f"{r['ci_lower']:>9.3f} "
            f"{r['ci_upper']:>9.3f} "
            f"{r['n_seeds']:>8}"
        )
    print("-" * 60)
    print(f"Pearson r  = {pr:+.3f}  (p = {pp:.4f})")
    print(f"Spearman ρ = {sr:+.3f}  (p = {sp:.4f})")
    print(f"Gradient recovered: {'YES ✓' if sr > 0 and sp < 0.10 else 'NO ✗'}")
    print()

    make_plot(rows, Path(args.output), show_wvs=args.wvs)


if __name__ == "__main__":
    main()
