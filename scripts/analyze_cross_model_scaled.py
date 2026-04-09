"""Scaled cross-model analysis — Section 3 of TOP_TIER_RESEARCH.md.

Phase 29.3 — GPT-4o-mini inverse effect robustness at N=50.

Loads existing N=20 single-seed results plus new N=50 multi-seed results,
computes bootstrap 95% CIs, tests whether the GPT-4o-mini inverse effect
(grounding *increases* bias) is statistically significant at larger scale.

Usage:
    python scripts/analyze_cross_model_scaled.py \
        --n20  analysis/cross_model_results.json \
        --n50  analysis/cross_model_results_n50.json \
        --out  analysis/cross_model_results_scaled.json

Outputs:
    analysis/cross_model_results_scaled.json   — combined table with CIs
    analysis/figures/cross_model_scaled_*.png  — comparison figures
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.analyze_padded_vs_grounded import mann_whitney_test, cohen_d


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Args:
        data: Observed values (one per seed or sample).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the CI.

    Raises:
        ValueError: If data is empty.
        IndexError: If data is empty (NumPy path).
    """
    if not data:
        raise ValueError("bootstrap_ci requires at least one data point.")

    arr = np.array(data, dtype=float)
    rng = np.random.default_rng(seed)

    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return lo, hi


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_seeded_results(
    rows: list[dict],
) -> dict[tuple[str, str], dict]:
    """Pool multi-seed results by (model_id, condition).

    Args:
        rows: List of result dicts, each with keys:
            model_id, condition, rlhf_bias_index, cooperation_rate, gini,
            n_agents, n_rounds.

    Returns:
        Dict keyed by (model_id, condition) with aggregated stats.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["model_id"], row["condition"])
        grouped.setdefault(key, []).append(row)

    agg: dict[tuple[str, str], dict] = {}
    for key, seed_rows in grouped.items():
        b_rlhf_vals = [r["rlhf_bias_index"] for r in seed_rows]
        coop_vals = [r["cooperation_rate"] for r in seed_rows]
        gini_vals = [r["gini"] for r in seed_rows]
        agg[key] = {
            "model_id": key[0],
            "condition": key[1],
            "n_seeds": len(seed_rows),
            "n_agents": seed_rows[0].get("n_agents", 0),
            "n_rounds": seed_rows[0].get("n_rounds", 0),
            "b_rlhf_values": b_rlhf_vals,
            "b_rlhf_mean": float(np.mean(b_rlhf_vals)),
            "b_rlhf_std": float(np.std(b_rlhf_vals)),
            "coop_rate_values": coop_vals,
            "coop_rate_mean": float(np.mean(coop_vals)),
            "gini_values": gini_vals,
            "gini_mean": float(np.mean(gini_vals)),
        }
    return agg


# ── Inverse effect significance ───────────────────────────────────────────────


def compute_inverse_effect_significance(
    b_rlhf_a: list[float],
    b_rlhf_b: list[float],
) -> dict:
    """Test whether the GPT-4o-mini inverse effect is statistically significant.

    The inverse effect occurs when B_RLHF(Condition B) > B_RLHF(Condition A):
    grounding *increases* the cooperation bias instead of reducing it.

    Args:
        b_rlhf_a: B_RLHF values for Condition A across seeds.
        b_rlhf_b: B_RLHF values for Condition B across seeds.

    Returns:
        Dict with: inverse_detected (bool), delta_mean (float),
        p_value (float), effect_size_d (float).
    """
    if not b_rlhf_a or not b_rlhf_b:
        return {
            "inverse_detected": False,
            "delta_mean": 0.0,
            "p_value": 1.0,
            "effect_size_d": 0.0,
            "note": "insufficient data",
        }

    mean_a = float(np.mean(b_rlhf_a))
    mean_b = float(np.mean(b_rlhf_b))
    delta = mean_b - mean_a  # positive = B higher than A = inverse

    # Statistical test only meaningful with ≥2 seeds per condition
    if len(b_rlhf_a) >= 2 and len(b_rlhf_b) >= 2:
        mw = mann_whitney_test(b_rlhf_a, b_rlhf_b)
        p_val = mw["p_value"]
        d = cohen_d(b_rlhf_b, b_rlhf_a)  # B - A direction
    else:
        p_val = 1.0  # not testable
        d = cohen_d(b_rlhf_b, b_rlhf_a) if len(b_rlhf_a) >= 2 else delta / 0.1

    # Inverse is "detected" if B > A and (significant OR large effect)
    inverse = delta > 0.02  # >2 percentage points difference

    return {
        "inverse_detected": bool(inverse),
        "delta_mean": round(float(delta), 4),
        "p_value": round(float(p_val), 4),
        "effect_size_d": round(float(d), 4) if not (
            isinstance(d, float) and (d != d)  # NaN check
        ) else 0.0,
    }


# ── Combined comparison table ─────────────────────────────────────────────────


def build_scaled_comparison_table(
    n20_results: list[dict],
    n50_results: list[dict],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Build unified comparison table across N=20 (old) and N=50 (new) results.

    Args:
        n20_results: Single-seed N=20 result rows (the original Phase 16 runs).
        n50_results: Multi-seed N=50 result rows (new scaled runs).
        n_bootstrap: Bootstrap resamples for CI computation.
        seed: Random seed for bootstrap.

    Returns:
        List of row dicts, one per model, with N=20 point estimates,
        N=50 means ± 95% CI, inverse effect flag, and significance test.
    """
    # N=20: single seed — just record point estimates
    n20_indexed: dict[tuple[str, str], dict] = {
        (r["model_id"], r["condition"]): r for r in n20_results
    }

    # N=50: aggregate across seeds
    n50_agg = aggregate_seeded_results(n50_results)

    # Collect all model IDs from both datasets
    all_models = sorted({r["model_id"] for r in n20_results + n50_results})

    rows = []
    for model in all_models:
        row: dict = {"model": model}

        # N=20 point estimates
        n20_a = n20_indexed.get((model, "A"), {})
        n20_b = n20_indexed.get((model, "B"), {})
        row["n20_bias_A"] = n20_a.get("rlhf_bias_index")
        row["n20_bias_B"] = n20_b.get("rlhf_bias_index")
        row["n20_coop_A"] = n20_a.get("cooperation_rate")
        row["n20_coop_B"] = n20_b.get("cooperation_rate")
        row["n20_n_agents"] = n20_a.get("n_agents", 20)

        # N=50 aggregated with CIs
        agg_a = n50_agg.get((model, "A"), {})
        agg_b = n50_agg.get((model, "B"), {})

        row["n50_bias_A_mean"] = agg_a.get("b_rlhf_mean")
        row["n50_bias_B_mean"] = agg_b.get("b_rlhf_mean")
        row["n50_coop_A_mean"] = agg_a.get("coop_rate_mean")
        row["n50_coop_B_mean"] = agg_b.get("coop_rate_mean")
        row["n50_n_seeds"] = agg_a.get("n_seeds", 0)
        row["n50_n_agents"] = agg_a.get("n_agents", 50)

        # Bootstrap CIs for N=50
        vals_a = agg_a.get("b_rlhf_values", [])
        vals_b = agg_b.get("b_rlhf_values", [])

        if vals_a:
            lo, hi = bootstrap_ci(vals_a, n_bootstrap=n_bootstrap, seed=seed)
            row["n50_bias_A_ci_lo"] = round(lo, 4)
            row["n50_bias_A_ci_hi"] = round(hi, 4)
        else:
            row["n50_bias_A_ci_lo"] = None
            row["n50_bias_A_ci_hi"] = None

        if vals_b:
            lo, hi = bootstrap_ci(vals_b, n_bootstrap=n_bootstrap, seed=seed + 1)
            row["n50_bias_B_ci_lo"] = round(lo, 4)
            row["n50_bias_B_ci_hi"] = round(hi, 4)
        else:
            row["n50_bias_B_ci_lo"] = None
            row["n50_bias_B_ci_hi"] = None

        # Inverse effect significance
        inv = compute_inverse_effect_significance(vals_a, vals_b)
        row["inverse_effect"] = inv["inverse_detected"]
        row["inverse_delta_mean"] = inv["delta_mean"]
        row["inverse_p_value"] = inv["p_value"]
        row["inverse_effect_size_d"] = inv["effect_size_d"]

        # Bias reduction (N=50)
        if agg_a.get("b_rlhf_mean") and agg_b.get("b_rlhf_mean") and agg_a["b_rlhf_mean"] > 0:
            delta_bias = agg_a["b_rlhf_mean"] - agg_b["b_rlhf_mean"]
            row["n50_bias_reduction_pct"] = round(delta_bias / agg_a["b_rlhf_mean"] * 100, 1)
        else:
            row["n50_bias_reduction_pct"] = None

        rows.append(row)

    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_scaled_comparison(table: list[dict], figures_dir: Path) -> None:
    """Generate comparison figures for the scaled cross-model results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in table]

    # B_RLHF comparison: N=20 vs N=50 for A and B conditions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models))
    width = 0.35

    for ax, condition, label in [
        (axes[0], "A", "Condition A (Ungrounded)"),
        (axes[1], "B", "Condition B (Grounded)"),
    ]:
        n20_vals = [r[f"n20_bias_{condition}"] or 0 for r in table]
        n50_vals = [r[f"n50_bias_{condition}_mean"] or 0 for r in table]
        n50_lo = [r[f"n50_bias_{condition}_ci_lo"] or 0 for r in table]
        n50_hi = [r[f"n50_bias_{condition}_ci_hi"] or 0 for r in table]
        n50_err = [[v - lo for v, lo in zip(n50_vals, n50_lo)],
                   [hi - v for v, hi in zip(n50_vals, n50_hi)]]

        ax.bar(x - width / 2, n20_vals, width, label="N=20 (single seed)",
               color="#95a5a6", alpha=0.8)
        ax.bar(x + width / 2, n50_vals, width, yerr=n50_err, capsize=5,
               label="N=50 (95% CI)", color="#3498db", alpha=0.85)

        # Annotate inverse effect
        for i, row in enumerate(table):
            if row["inverse_effect"]:
                ax.annotate("*inverse*", xy=(x[i] + width / 2, n50_vals[i] + 0.02),
                            ha="center", fontsize=8, color="red")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=9)
        ax.set_ylabel("B_RLHF")
        ax.set_title(f"{label}\nB_RLHF by Scale")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 0.7)

    fig.suptitle("Cross-Model Validation: N=20 vs N=50", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = figures_dir / "cross_model_scaled_bias.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # GPT-4o-mini inverse effect focus plot
    gpt_rows = [r for r in table if "gpt" in r["model"].lower()]
    if gpt_rows:
        gpt = gpt_rows[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        conditions = ["A (Ungrounded)", "B (Grounded)"]
        n20 = [gpt["n20_bias_A"] or 0, gpt["n20_bias_B"] or 0]
        n50 = [gpt["n50_bias_A_mean"] or 0, gpt["n50_bias_B_mean"] or 0]
        n50_lo = [gpt["n50_bias_A_ci_lo"] or 0, gpt["n50_bias_B_ci_lo"] or 0]
        n50_hi = [gpt["n50_bias_A_ci_hi"] or 0, gpt["n50_bias_B_ci_hi"] or 0]
        n50_err = [[v - lo for v, lo in zip(n50, n50_lo)],
                   [hi - v for v, hi in zip(n50, n50_hi)]]

        xpos = np.arange(2)
        ax.bar(xpos - 0.2, n20, 0.35, label="N=20", color="#e74c3c", alpha=0.8)
        ax.bar(xpos + 0.2, n50, 0.35, yerr=n50_err, capsize=5,
               label="N=50 (95% CI)", color="#e67e22", alpha=0.85)
        ax.set_xticks(xpos)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("B_RLHF")
        inverse_str = "SIGNIFICANT" if gpt.get("inverse_p_value", 1) < 0.05 else "marginal"
        ax.set_title(f"GPT-4o-mini: Inverse Grounding Effect ({inverse_str})\n"
                     f"p={gpt.get('inverse_p_value', 'N/A')}, "
                     f"d={gpt.get('inverse_effect_size_d', 'N/A')}")
        ax.legend()
        plt.tight_layout()
        out = figures_dir / "cross_model_gpt4o_inverse.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scaled cross-model analysis with bootstrap CIs."
    )
    parser.add_argument(
        "--n20", type=str,
        default="analysis/cross_model_results.json",
        help="Existing N=20 results JSON.",
    )
    parser.add_argument(
        "--n50", type=str,
        default="analysis/cross_model_results_n50.json",
        help="New N=50 multi-seed results JSON.",
    )
    parser.add_argument(
        "--out", type=str,
        default="analysis/cross_model_results_scaled.json",
        help="Output path for combined table.",
    )
    parser.add_argument(
        "--figures-dir", type=str,
        default="analysis/figures",
        help="Output directory for figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n20_path = Path(args.n20)
    n50_path = Path(args.n50)

    n20_results = json.loads(n20_path.read_text()) if n20_path.exists() else []
    n50_results = json.loads(n50_path.read_text()) if n50_path.exists() else []

    if not n20_results and not n50_results:
        print("No results found. Run run_cross_model_comparison.py --seeds first.")
        return

    print(f"Loaded {len(n20_results)} N=20 rows, {len(n50_results)} N=50 rows.")

    table = build_scaled_comparison_table(n20_results, n50_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"Scaled comparison table saved to: {out_path}")

    print("\n── Cross-Model Scaled Results (Table 3) ────────────────────")
    print(f"{'Model':<20} {'N20 A':>7} {'N20 B':>7} {'N50 A':>7} {'N50 B':>7} {'Inverse':>8}")
    print("-" * 60)
    for row in table:
        inv_flag = "YES *" if row["inverse_effect"] else "no"
        print(
            f"{row['model']:<20} "
            f"{row['n20_bias_A'] or 'N/A':>7} "
            f"{row['n20_bias_B'] or 'N/A':>7} "
            f"{row['n50_bias_A_mean'] or 'N/A':>7} "
            f"{row['n50_bias_B_mean'] or 'N/A':>7} "
            f"{inv_flag:>8}"
        )

    plot_scaled_comparison(table, Path(args.figures_dir))


if __name__ == "__main__":
    main()
