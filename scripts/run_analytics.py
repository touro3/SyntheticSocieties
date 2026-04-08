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
    parser.add_argument(
        "--experiment-ids",
        default="",
        help="Optional comma-separated experiment IDs to include.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seeds to include.",
    )
    parser.add_argument(
        "--policy-types",
        default="",
        help="Optional comma-separated policy types to include.",
    )
    parser.add_argument(
        "--require-cmp-only",
        action="store_true",
        help="Restrict analytics to experiment_id LIKE 'cmp_%%'.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BGF Analytics — DuckDB Queries")
    print("=" * 60)

    experiment_ids = [x.strip() for x in args.experiment_ids.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    policy_types = [x.strip() for x in args.policy_types.split(",") if x.strip()]

    if experiment_ids:
        print(f"Scope: {len(experiment_ids)} experiment IDs")
    if seeds:
        print(f"Scope seeds: {seeds}")
    if policy_types:
        print(f"Scope policies: {policy_types}")
    if args.require_cmp_only:
        print("Scope: cmp_* only")

    run_all_queries(
        index_path=args.index,
        output_dir=args.output_dir,
        experiment_ids=experiment_ids or None,
        seeds=seeds or None,
        policy_types=policy_types or None,
        require_cmp_only=args.require_cmp_only,
    )


if __name__ == "__main__":
    main()
