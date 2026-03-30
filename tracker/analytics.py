"""
DuckDB analytics queries over experiment_index.parquet.

Provides structured queries for analyzing experiment results:
  - Policy comparison
  - Seed variance analysis
  - Ablation comparison
  - LLM vs baseline comparison
  - Robustness summary
  - Statistical significance (Mann-Whitney U, Cohen's d, confidence intervals)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


DEFAULT_INDEX = "tracker/experiment_index.parquet"


def _connect(index_path: str = DEFAULT_INDEX) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the experiment index loaded."""
    conn = duckdb.connect()
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment index not found: {path}")
    conn.execute(f"CREATE VIEW experiments AS SELECT * FROM read_parquet('{path}')")
    return conn


def query_by_policy(index_path: str = DEFAULT_INDEX) -> pd.DataFrame:
    """Aggregate metrics by policy type."""
    conn = _connect(index_path)
    return conn.execute("""
        SELECT
            policy_type,
            COUNT(*) as n_experiments,
            AVG(wealth_mean) as avg_wealth_mean,
            STDDEV(wealth_mean) as std_wealth_mean,
            AVG(wealth_gini) as avg_gini,
            AVG(stress_mean) as avg_stress_mean
        FROM experiments
        GROUP BY policy_type
        ORDER BY avg_wealth_mean DESC
    """).fetchdf()


def query_by_seed(
    policy: str = "llm",
    index_path: str = DEFAULT_INDEX,
) -> pd.DataFrame:
    """Variance analysis across seeds for a given policy."""
    conn = _connect(index_path)
    return conn.execute(f"""
        SELECT
            experiment_id,
            seed,
            wealth_mean,
            wealth_gini,
            stress_mean
        FROM experiments
        WHERE policy_type = '{policy}'
        ORDER BY seed
    """).fetchdf()


def query_by_ablation(index_path: str = DEFAULT_INDEX) -> pd.DataFrame:
    """Compare ablation conditions (experiments with 'ablation_' prefix)."""
    conn = _connect(index_path)
    return conn.execute("""
        SELECT
            REGEXP_EXTRACT(experiment_id, 'ablation_([^_]+(?:_[^_]+)?)', 1) as ablation_mode,
            COUNT(*) as n_runs,
            AVG(wealth_mean) as avg_wealth_mean,
            STDDEV(wealth_mean) as std_wealth_mean,
            AVG(wealth_gini) as avg_gini,
            AVG(stress_mean) as avg_stress_mean
        FROM experiments
        WHERE experiment_id LIKE 'ablation_%'
        GROUP BY ablation_mode
        ORDER BY avg_wealth_mean DESC
    """).fetchdf()


def compare_llm_vs_baselines(index_path: str = DEFAULT_INDEX) -> pd.DataFrame:
    """Head-to-head comparison: LLM, ablated_llm, template, rule_based, random."""
    conn = _connect(index_path)
    return conn.execute("""
        SELECT
            policy_type,
            COUNT(*) as n_runs,
            AVG(wealth_mean) as avg_wealth_mean,
            STDDEV(wealth_mean) as std_wealth_mean,
            AVG(wealth_gini) as avg_gini,
            MIN(wealth_mean) as min_wealth_mean,
            MAX(wealth_mean) as max_wealth_mean
        FROM experiments
        GROUP BY policy_type
        ORDER BY
            CASE policy_type
                WHEN 'llm' THEN 1
                WHEN 'ablated_llm' THEN 2
                WHEN 'template' THEN 3
                WHEN 'data_driven' THEN 4
                WHEN 'rule_based' THEN 5
                WHEN 'random' THEN 6
                WHEN 'mock' THEN 7
                ELSE 8
            END
    """).fetchdf()


def robustness_summary(index_path: str = DEFAULT_INDEX) -> dict[str, pd.DataFrame]:
    """Generate robustness analysis tables."""
    conn = _connect(index_path)

    # Temperature sensitivity
    temp = conn.execute("""
        SELECT
            experiment_id,
            wealth_mean,
            wealth_gini,
            stress_mean
        FROM experiments
        WHERE experiment_id LIKE 'llm_temp_%'
        ORDER BY experiment_id
    """).fetchdf()

    # Horizon sweep
    horizon = conn.execute("""
        SELECT
            experiment_id,
            wealth_mean,
            wealth_gini,
            stress_mean
        FROM experiments
        WHERE experiment_id LIKE 'llm_horizon_%'
        ORDER BY experiment_id
    """).fetchdf()

    # Seed variance
    seed_var = conn.execute("""
        SELECT
            policy_type,
            STDDEV(wealth_mean) / NULLIF(AVG(wealth_mean), 0) as cv_wealth,
            STDDEV(wealth_gini) / NULLIF(AVG(wealth_gini), 0) as cv_gini
        FROM experiments
        WHERE experiment_id LIKE 'cmp_%'
        GROUP BY policy_type
    """).fetchdf()

    return {
        "temperature_sensitivity": temp,
        "horizon_sweep": horizon,
        "seed_variance": seed_var,
    }


# ── Statistical Significance ─────────────────────────────────────────────────


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns 0.0 when both groups have
    zero variance to avoid division by zero.
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def mann_whitney_test(
    group_a: np.ndarray, group_b: np.ndarray
) -> dict[str, float]:
    """Run a two-sided Mann-Whitney U test.

    Returns dict with keys: U_statistic, p_value.
    Returns p_value=1.0 for degenerate inputs.
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    if len(group_a) < 1 or len(group_b) < 1:
        return {"U_statistic": 0.0, "p_value": 1.0}
    u_stat, p_val = scipy_stats.mannwhitneyu(
        group_a, group_b, alternative="two-sided"
    )
    return {"U_statistic": float(u_stat), "p_value": float(p_val)}


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Returns dict with keys: estimate, ci_lower, ci_upper, confidence.
    """
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "confidence": confidence}
    rng = np.random.RandomState(seed)
    estimate = float(statistic_fn(data))
    if len(data) == 1:
        return {"estimate": estimate, "ci_lower": estimate, "ci_upper": estimate, "confidence": confidence}
    boot_stats = np.array([
        statistic_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
    }


def fdr_correct(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Returns adjusted p-values (same length, same order as input).
    Adjusted values are capped at 1.0.
    """
    if not p_values:
        return []
    m = len(p_values)
    if m == 1:
        return list(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = m - rank_from_end  # 1-based rank from smallest
        adj = min(prev, p * m / rank)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


def pairwise_significance(
    df: pd.DataFrame,
    metric_col: str = "wealth_mean",
    group_col: str = "policy_type",
    reference_group: str = "llm",
) -> pd.DataFrame:
    """Compare each group against a reference group with Mann-Whitney U + Cohen's d.

    Parameters
    ----------
    df : DataFrame with per-experiment rows (one row per seed/run).
    metric_col : Column to compare.
    group_col : Column identifying groups.
    reference_group : The group to compare all others against.

    Returns
    -------
    DataFrame with columns: group, ref_mean, group_mean, cohens_d,
    U_statistic, p_value, significant_005, p_value_fdr, significant_005_fdr.
    """
    ref_vals = df.loc[df[group_col] == reference_group, metric_col].values
    if len(ref_vals) == 0:
        return pd.DataFrame()

    rows = []
    for group_name, group_df in df.groupby(group_col):
        if group_name == reference_group:
            continue
        group_vals = group_df[metric_col].values
        mw = mann_whitney_test(ref_vals, group_vals)
        d = cohens_d(ref_vals, group_vals)
        rows.append({
            "group": group_name,
            "ref_mean": float(np.mean(ref_vals)),
            "group_mean": float(np.mean(group_vals)),
            "cohens_d": d,
            "U_statistic": mw["U_statistic"],
            "p_value": mw["p_value"],
            "significant_005": mw["p_value"] < 0.05,
        })

    if not rows:
        return pd.DataFrame()

    # Apply FDR correction across all comparisons
    raw_p = [r["p_value"] for r in rows]
    corrected_p = fdr_correct(raw_p)
    for row, fdr_p in zip(rows, corrected_p):
        row["p_value_fdr"] = fdr_p
        row["significant_005_fdr"] = fdr_p < 0.05

    return pd.DataFrame(rows)


def run_all_queries(
    index_path: str = DEFAULT_INDEX,
    output_dir: str = "analysis/tables",
) -> None:
    """Run all analytics queries and save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Running DuckDB analytics...")

    try:
        df = query_by_policy(index_path)
        df.to_csv(out / "policy_comparison.csv", index=False)
        print(f"  ✓ policy_comparison.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ policy_comparison: {e}")

    try:
        df = compare_llm_vs_baselines(index_path)
        df.to_csv(out / "llm_vs_baselines.csv", index=False)
        print(f"  ✓ llm_vs_baselines.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ llm_vs_baselines: {e}")

    try:
        df = query_by_ablation(index_path)
        df.to_csv(out / "ablation_comparison.csv", index=False)
        print(f"  ✓ ablation_comparison.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ ablation_comparison: {e}")

    try:
        tables = robustness_summary(index_path)
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"  ✓ {name}.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ robustness: {e}")

    print("Done!")
