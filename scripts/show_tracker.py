import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl


def main() -> None:
    tracker_path = Path("tracker/experiment_index.parquet")

    if not tracker_path.exists():
        print("Tracker file does not exist yet.")
        return

    df = pl.read_parquet(tracker_path)
    print(df)


if __name__ == "__main__":
    main()
