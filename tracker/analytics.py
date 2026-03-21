"""
DuckDB analytics queries over experiment_index.parquet.

Provides structured queries for analyzing experiment results:
  - Policy comparison
  - Seed variance analysis
  - Ablation comparison
  - LLM vs baseline comparison
  - Robustness summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


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
