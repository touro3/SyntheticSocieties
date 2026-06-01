import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.tracker import append_record, build_experiment_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register one experiment into the BGF tracker.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to experiment directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_experiment_record(args.run_dir)
    append_record(record)
    print(f"Registered experiment: {record['experiment_id']}")


if __name__ == "__main__":
    main()
