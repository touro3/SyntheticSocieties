from pathlib import Path

import polars as pl


def main():

    tracker = pl.read_parquet("tracker/experiment_index.parquet")

    table = (
        tracker.group_by(["policy_type", "network_type"])
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
        .sort(["policy_type", "network_type"])
    )

    Path("analysis/tables").mkdir(parents=True, exist_ok=True)

    table.write_csv("analysis/tables/policy_network_comparison.csv")
    table.write_parquet("analysis/tables/policy_network_comparison.parquet")

    print(table)


if __name__ == "__main__":
    main()
