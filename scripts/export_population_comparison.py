from pathlib import Path

import polars as pl


def main() -> None:
    tracker_path = Path("tracker/experiment_index.parquet")
    output_csv = Path("analysis/tables/population_comparison.csv")
    output_parquet = Path("analysis/tables/population_comparison.parquet")

    if not tracker_path.exists():
        print("Tracker file does not exist.")
        return

    df = pl.read_parquet(tracker_path)

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(output_csv)
    summary.write_parquet(output_parquet)

    print("Population comparison table created.")
    print(summary)
    print(f"CSV saved to: {output_csv}")
    print(f"Parquet saved to: {output_parquet}")


if __name__ == "__main__":
    main()
