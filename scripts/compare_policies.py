import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl


def main() -> None:
    tracker_path = Path("tracker/experiment_index.parquet")

    if not tracker_path.exists():
        print("Tracker file does not exist.")
        return

    df = pl.read_parquet(tracker_path)

    summary = (
        df.group_by("policy_type")
        .agg(
            [
                pl.col("wealth_mean").mean().alias("avg_wealth_mean"),
                pl.col("wealth_gini").mean().alias("avg_wealth_gini"),
                pl.col("stress_mean").mean().alias("avg_stress_mean"),
            ]
        )
        .sort("policy_type")
    )

    print(summary)


if __name__ == "__main__":
    main()