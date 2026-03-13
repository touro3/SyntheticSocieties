import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl


def test_policy_aggregation_logic():
    df = pl.DataFrame(
        {
            "policy_type": ["mock", "mock", "random", "random"],
            "wealth_mean": [80.0, 82.0, 90.0, 86.0],
            "wealth_gini": [0.05, 0.06, 0.10, 0.12],
            "stress_mean": [0.7, 0.9, 1.1, 1.3],
        }
    )

    summary = (
        df.group_by("policy_type")
        .agg(
            [
                pl.len().alias("n_runs"),
                pl.col("wealth_mean").mean().alias("wealth_mean_avg"),
                pl.col("wealth_mean").std().alias("wealth_mean_std"),
                pl.col("wealth_gini").mean().alias("wealth_gini_avg"),
                pl.col("wealth_gini").std().alias("wealth_gini_std"),
                pl.col("stress_mean").mean().alias("stress_mean_avg"),
                pl.col("stress_mean").std().alias("stress_mean_std"),
            ]
        )
        .sort("policy_type")
    )

    random_row = summary.filter(pl.col("policy_type") == "random").to_dicts()[0]

    assert random_row["n_runs"] == 2
    assert random_row["wealth_mean_avg"] == 88.0
    assert abs(random_row["stress_mean_avg"] - 1.2) < 1e-9
