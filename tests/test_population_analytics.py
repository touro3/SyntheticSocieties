import polars as pl


def test_population_aggregation_logic():
    df = pl.DataFrame(
        {
            "policy_type": ["mock", "mock", "mock", "random"],
            "population_size": [20, 20, 50, 20],
            "wealth_mean": [80.0, 82.0, 90.0, 85.0],
            "wealth_gini": [0.05, 0.06, 0.09, 0.10],
            "stress_mean": [0.7, 0.9, 1.2, 1.1],
        }
    )

    summary = (
        df.group_by(["policy_type", "population_size"])
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
        .sort(["policy_type", "population_size"])
    )

    row = summary.filter(
        (pl.col("policy_type") == "mock") & (pl.col("population_size") == 20)
    ).to_dicts()[0]

    assert row["n_runs"] == 2
    assert row["wealth_mean_avg"] == 81.0
    assert abs(row["stress_mean_avg"] - 0.8) < 1e-9