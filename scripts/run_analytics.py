"""
Run DuckDB analytics over experiment results.

Usage:
    python scripts/run_analytics.py
    python scripts/run_analytics.py --index tracker/experiment_index.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tracker.analytics import run_all_queries


def main():
    parser = argparse.ArgumentParser(description="Run BGF analytics queries.")
    parser.add_argument(
        "--index",
        default="tracker/experiment_index.parquet",
        help="Path to experiment index parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/tables",
        help="Output directory for CSV tables.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BGF Analytics — DuckDB Queries")
    print("=" * 60)

    run_all_queries(
        index_path=args.index,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
