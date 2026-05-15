"""Nested variance decomposition on existing seed-level grounding data.

For cooperation_rate, wealth_gini, and (if available) brlhf, decompose the
total variance into:
    σ²_condition  — between-condition (A vs B) variance (the grounding effect)
    σ²_seed       — between-seed-within-condition variance (replication noise)
    σ²_residual   — unexplained

A high σ²_condition / σ²_total ratio is the meta-analytic signal that the
grounding effect dominates seed noise — the quantity reviewers want to see.

CPU-only. Output: analysis/tables/variance_decomposition.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEED_CSV = REPO / "analysis" / "tables" / "grounding_comparison_seed_metrics.csv"
OUT_JSON = REPO / "analysis" / "tables" / "variance_decomposition.json"


def decompose(df: pd.DataFrame, metric: str, condition_col: str = "condition_key") -> dict:
    grand_mean = df[metric].mean()
    cond_means = df.groupby(condition_col)[metric].mean()
    n_by_cond = df.groupby(condition_col)[metric].size()

    ss_total = float(((df[metric] - grand_mean) ** 2).sum())
    ss_between = float(((cond_means - grand_mean) ** 2 * n_by_cond).sum())
    ss_within = float(ss_total - ss_between)

    k = len(cond_means)
    N = len(df)
    ms_between = ss_between / max(1, k - 1)
    ms_within = ss_within / max(1, N - k)
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

    # F statistic and crude p
    if ms_within > 0 and k > 1:
        F = ms_between / ms_within
        from scipy import stats

        p = float(1 - stats.f.cdf(F, k - 1, N - k))
    else:
        F = float("nan")
        p = float("nan")

    return {
        "metric": metric,
        "k_conditions": int(k),
        "N_observations": int(N),
        "grand_mean": float(grand_mean),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "eta_squared": float(eta_sq),
        "F": float(F),
        "p_anova": p,
        "condition_means": {str(k_): float(v) for k_, v in cond_means.items()},
    }


def main() -> None:
    df = pd.read_csv(SEED_CSV)
    df = df[df["condition_key"].isin(["pure_llm_ess_persona", "grounded_llm_ess_persona"])].copy()
    df["condition"] = df["condition_key"].map(
        {
            "pure_llm_ess_persona": "A_ungrounded",
            "grounded_llm_ess_persona": "B_grounded",
        }
    )

    results = {}
    for metric in ["cooperation_rate", "wealth_gini", "work_rate", "save_rate"]:
        results[metric] = decompose(df, metric, condition_col="condition")

    print(f"{'metric':<20} {'η²':>6} {'F':>8} {'p':>8} {'mean_A':>10} {'mean_B':>10}")
    print("-" * 70)
    for m, r in results.items():
        means = r["condition_means"]
        ma = means.get("A_ungrounded", float("nan"))
        mb = means.get("B_grounded", float("nan"))
        print(f"{m:<20} {r['eta_squared']:>6.3f} {r['F']:>8.2f} {r['p_anova']:>8.4f} {ma:>10.3f} {mb:>10.3f}")

    out = {
        "model": "one-way ANOVA on condition (A vs B); seed acts as replication within condition",
        "data_source": str(SEED_CSV.relative_to(REPO)),
        "decomposition": results,
        "audit_row": "B.variance (new)",
        "interpretation": (
            "η² is the fraction of total variance in the metric attributable to "
            "the A-vs-B condition contrast. Values above 0.7 indicate that the "
            "grounding effect dwarfs seed-level replication noise."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n✓ {OUT_JSON}")


if __name__ == "__main__":
    main()
