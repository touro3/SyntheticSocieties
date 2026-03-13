from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from utils.config import load_config
from utils.io import ensure_dir
import json


TRACKER_PATH = Path("tracker/experiment_index.parquet")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_record(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)

    metadata = load_json(run_dir / "metadata.json")
    summary = load_json(run_dir / "summary.json")
    config = load_config(run_dir / "config.yaml")

    return {
        "experiment_id": metadata["experiment_id"],
        "project_name": metadata["project_name"],
        "seed": metadata["seed"],
        "policy_type": metadata["policy_type"],
        "population_size": metadata["population_size"],
        "rounds": metadata["rounds"],

        "network_type": metadata.get("network_type", config["network"]["type"]),
        "network_edge_prob": metadata.get("network_edge_prob", config["network"].get("edge_prob")),

        "wealth_mean": summary["wealth"]["mean"],
        "wealth_median": summary["wealth"]["median"],
        "wealth_variance": summary["wealth"]["variance"],
        "wealth_gini": summary["wealth"]["gini"],
        "wealth_min": summary["wealth"]["min"],
        "wealth_max": summary["wealth"]["max"],
        "stress_mean": summary["stress"]["mean"],
        "stress_variance": summary["stress"]["variance"],
        "num_agents": summary["num_agents"],
        "num_work": summary.get("event_action_counts", {}).get("work", 0),
        "num_save": summary.get("event_action_counts", {}).get("save", 0),
        "num_cooperate": summary.get("event_action_counts", {}).get("cooperate", 0),

        "config_path": str(run_dir / "config.yaml"),
        "metadata_path": str(run_dir / "metadata.json"),
        "summary_path": str(run_dir / "summary.json"),
        "events_path": str(run_dir / "events.jsonl"),
        "environment_economy": config["environment"]["public_signal"].get("economy", "unknown"),
    }


def append_record(record: dict[str, Any], tracker_path: str | Path = TRACKER_PATH) -> None:
    tracker_path = Path(tracker_path)
    ensure_dir(tracker_path.parent)

    new_df = pl.DataFrame([record])

    if tracker_path.exists():
        existing = pl.read_parquet(tracker_path)

        if "experiment_id" in existing.columns and record["experiment_id"] in existing["experiment_id"].to_list():
            existing = existing.filter(pl.col("experiment_id") != record["experiment_id"])

        updated = pl.concat([existing, new_df], how="diagonal")
    else:
        updated = new_df

    updated.write_parquet(tracker_path)


def rebuild_tracker(experiments_dir: str | Path = "experiments", tracker_path: str | Path = TRACKER_PATH) -> None:
    experiments_dir = Path(experiments_dir)
    tracker_path = Path(tracker_path)

    records: list[dict[str, Any]] = []

    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        required = [
            run_dir / "config.yaml",
            run_dir / "metadata.json",
            run_dir / "summary.json",
        ]

        if all(path.exists() for path in required):
            records.append(build_experiment_record(run_dir))

    ensure_dir(tracker_path.parent)

    if records:
        df = pl.DataFrame(records)
        df.write_parquet(tracker_path)
    else:
        pl.DataFrame().write_parquet(tracker_path)