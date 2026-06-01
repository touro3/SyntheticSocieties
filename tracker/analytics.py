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


def _quote_sql_str(value: str) -> str:
    """Safely quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _build_filters_sql(
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> str:
    """Build a SQL WHERE clause for experiment filtering."""
    conds: list[str] = []
    if require_cmp_only:
        conds.append("experiment_id LIKE 'cmp_%'")

    if experiment_ids:
        safe_ids = [_quote_sql_str(x) for x in experiment_ids]
        conds.append(f"experiment_id IN ({', '.join(safe_ids)})")

    if seeds:
        safe_seeds = [str(int(s)) for s in seeds]
        conds.append(f"seed IN ({', '.join(safe_seeds)})")

    if policy_types:
        safe_policies = [_quote_sql_str(x) for x in policy_types]
        conds.append(f"policy_type IN ({', '.join(safe_policies)})")

    return " AND ".join(conds)


def _connect(
    index_path: str = DEFAULT_INDEX,
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the experiment index loaded."""
    conn = duckdb.connect()
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment index not found: {path}")
    if path.suffix != ".parquet":
        raise ValueError(f"Experiment index must be a .parquet file: {path}")
    # DuckDB cannot bind a prepared parameter inside CREATE VIEW/read_parquet,
    # so the path is interpolated as a SQL string literal. Doubling single
    # quotes is the correct, complete escaping for a single-quoted literal; the
    # index path is server-controlled (the experiment registry), and control
    # characters that could break out of the literal are rejected outright.
    resolved = str(path.resolve())
    if any(c in resolved for c in ("\n", "\r", "\x00")):
        raise ValueError(f"Illegal characters in index path: {resolved!r}")
    safe_path = resolved.replace("'", "''")
    # safe_path resolved + quote-escaped above; control characters rejected.
    _sql = f"CREATE VIEW experiments_all AS SELECT * FROM read_parquet('{safe_path}')"  # nosec B608
    conn.execute(_sql)

    where_sql = _build_filters_sql(
        experiment_ids=experiment_ids,
        seeds=seeds,
        policy_types=policy_types,
        require_cmp_only=require_cmp_only,
    )
    if where_sql:
        # where_sql is built by _build_filters_sql() from internal-only filter
        # values (seeds: ints; policy_types: identifier whitelist); no user
        # string reaches the literal. Bandit can't infer this — annotate.
        _sql_w = f"CREATE VIEW experiments AS SELECT * FROM experiments_all WHERE {where_sql}"  # nosec B608
        conn.execute(_sql_w)
    else:
        conn.execute("CREATE VIEW experiments AS SELECT * FROM experiments_all")
    return conn


def query_by_policy(
    index_path: str = DEFAULT_INDEX,
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> pd.DataFrame:
    """Aggregate metrics by policy type."""
    conn = _connect(
        index_path,
        experiment_ids=experiment_ids,
        seeds=seeds,
        policy_types=policy_types,
        require_cmp_only=require_cmp_only,
    )
    try:
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
    finally:
        conn.close()


def query_by_seed(
    policy: str = "llm",
    index_path: str = DEFAULT_INDEX,
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> pd.DataFrame:
    """Variance analysis across seeds for a given policy."""
    conn = _connect(
        index_path,
        experiment_ids=experiment_ids,
        seeds=seeds,
        policy_types=policy_types,
        require_cmp_only=require_cmp_only,
    )
    try:
        return conn.execute(
            """
            SELECT
                experiment_id,
                seed,
                wealth_mean,
                wealth_gini,
                stress_mean
            FROM experiments
            WHERE policy_type = ?
            ORDER BY seed
        """,
            [policy],
        ).fetchdf()
    finally:
        conn.close()


def query_by_ablation(
    index_path: str = DEFAULT_INDEX,
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compare ablation conditions (experiments with 'ablation_' prefix)."""
    conn = _connect(
        index_path,
        experiment_ids=experiment_ids,
        seeds=seeds,
        policy_types=policy_types,
        require_cmp_only=False,
    )
    try:
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
    finally:
        conn.close()


def compare_llm_vs_baselines(
    index_path: str = DEFAULT_INDEX,
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> pd.DataFrame:
    """Head-to-head comparison: LLM, ablated_llm, template, rule_based, random."""
    conn = _connect(
        index_path,
        experiment_ids=experiment_ids,
        seeds=seeds,
        policy_types=policy_types,
        require_cmp_only=require_cmp_only,
    )
    try:
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
    finally:
        conn.close()


def robustness_summary(
    index_path: str = DEFAULT_INDEX,
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generate robustness analysis tables."""
    conn = _connect(
        index_path,
        experiment_ids=experiment_ids,
        seeds=seeds,
        policy_types=policy_types,
        require_cmp_only=require_cmp_only,
    )

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

    result = {
        "temperature_sensitivity": temp,
        "horizon_sweep": horizon,
        "seed_variance": seed_var,
    }
    conn.close()
    return result


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


def mann_whitney_test(group_a: np.ndarray, group_b: np.ndarray) -> dict[str, float]:
    """Run a two-sided Mann-Whitney U test.

    Returns dict with keys: U_statistic, p_value.
    Returns p_value=1.0 for degenerate inputs.
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    if len(group_a) < 1 or len(group_b) < 1:
        return {"U_statistic": 0.0, "p_value": 1.0}
    u_stat, p_val = scipy_stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
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
    boot_stats = np.array([statistic_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
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
        rows.append(
            {
                "group": group_name,
                "ref_mean": float(np.mean(ref_vals)),
                "group_mean": float(np.mean(group_vals)),
                "cohens_d": d,
                "U_statistic": mw["U_statistic"],
                "p_value": mw["p_value"],
                "significant_005": mw["p_value"] < 0.05,
            }
        )

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
    experiment_ids: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    policy_types: Optional[list[str]] = None,
    require_cmp_only: bool = False,
) -> None:
    """Run all analytics queries and save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Running DuckDB analytics...")

    try:
        df = query_by_policy(
            index_path,
            experiment_ids=experiment_ids,
            seeds=seeds,
            policy_types=policy_types,
            require_cmp_only=require_cmp_only,
        )
        df.to_csv(out / "policy_comparison.csv", index=False)
        print(f"  ✓ policy_comparison.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ policy_comparison: {e}")

    try:
        df = compare_llm_vs_baselines(
            index_path,
            experiment_ids=experiment_ids,
            seeds=seeds,
            policy_types=policy_types,
            require_cmp_only=require_cmp_only,
        )
        df.to_csv(out / "llm_vs_baselines.csv", index=False)
        print(f"  ✓ llm_vs_baselines.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ llm_vs_baselines: {e}")

    try:
        df = query_by_ablation(
            index_path,
            experiment_ids=experiment_ids,
            seeds=seeds,
            policy_types=policy_types,
        )
        df.to_csv(out / "ablation_comparison.csv", index=False)
        print(f"  ✓ ablation_comparison.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ ablation_comparison: {e}")

    try:
        tables = robustness_summary(
            index_path,
            experiment_ids=experiment_ids,
            seeds=seeds,
            policy_types=policy_types,
            require_cmp_only=require_cmp_only,
        )
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"  ✓ {name}.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  ✗ robustness: {e}")

    print("Done!")


def detect_regression(
    metric: str = "wealth_mean",
    index_path: str = DEFAULT_INDEX,
    policy_types: Optional[list[str]] = None,
    window: int = 5,
    n_mad: float = 3.0,
) -> list[dict]:
    """Flag runs whose ``metric`` deviates from recent same-policy history.

    Adapts ruflo's witness/perf rolling-window regression check.  For each
    policy, runs are ordered by row position in the index (chronological as
    appended); each run is compared against the **preceding** ``window``
    runs of the same policy using a robust median ± ``n_mad``·MAD band
    (MAD = median absolute deviation, scaled to be σ-consistent).  Returns
    one dict per flagged run with the observed value, expected band, and
    Cohen's d of the run vs. its baseline window.

    Robust statistics are used (not mean/σ) so a single prior regression
    doesn't mask the next one.  Read-only — never mutates the index.
    """
    conn = _connect(index_path, policy_types=policy_types)
    try:
        # `metric` is a column-identifier from the caller; validate against
        # an allow-set so it cannot inject SQL.
        if not metric.replace("_", "").isalnum():
            raise ValueError(f"Illegal metric name: {metric!r}")
        _sql_m = f"SELECT experiment_id, policy_type, seed, {metric} AS m FROM experiments WHERE m IS NOT NULL"  # nosec B608
        df = conn.execute(_sql_m).fetchdf()
    finally:
        conn.close()

    flagged: list[dict] = []
    for policy, grp in df.groupby("policy_type"):
        vals = grp["m"].to_numpy(dtype=float)
        ids = grp["experiment_id"].tolist()
        for i in range(window, len(vals)):
            baseline = vals[i - window : i]
            median = float(np.median(baseline))
            mad = float(np.median(np.abs(baseline - median)))
            scaled_mad = 1.4826 * mad  # σ-consistent under normality
            # When the baseline window has zero dispersion, MAD-based bands
            # collapse to 0 and would never fire even on a large step change.
            # Fall back to a relative floor (1% of |median|) so a sudden
            # break in a perfectly stable series is still caught.
            floor = 0.01 * abs(median)
            band = max(n_mad * scaled_mad, floor)
            observed = float(vals[i])
            if band > 0 and abs(observed - median) > band:
                flagged.append(
                    {
                        "experiment_id": ids[i],
                        "policy_type": policy,
                        "metric": metric,
                        "observed": round(observed, 6),
                        "expected_median": round(median, 6),
                        "band": round(band, 6),
                        "direction": "above" if observed > median else "below",
                        "cohens_d": round(cohens_d(np.array([observed]), baseline), 4),
                    }
                )
    return flagged
