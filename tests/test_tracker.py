import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.io import ensure_dir, save_json, save_yaml
from utils.tracker import build_experiment_record


def test_build_experiment_record(tmp_path):
    run_dir = ensure_dir(tmp_path / "exp_test")

    save_yaml(
        {
            "environment": {
                "public_signal": {"economy": "stable"},
            }
        },
        run_dir / "config.yaml",
    )

    save_json(
        {
            "project_name": "bgf",
            "experiment_id": "exp_test",
            "seed": 42,
            "policy_type": "mock",
            "population_size": 5,
            "rounds": 3,
        },
        run_dir / "metadata.json",
    )

    save_json(
        {
            "num_agents": 5,
            "wealth": {
                "mean": 100.0,
                "median": 100.0,
                "variance": 10.0,
                "gini": 0.1,
                "min": 90.0,
                "max": 110.0,
            },
            "stress": {
                "mean": 2.0,
                "variance": 0.5,
            },
            "actions": {
                "work": 4,
                "save": 1,
            },
        },
        run_dir / "summary.json",
    )

    record = build_experiment_record(run_dir)

    assert record["experiment_id"] == "exp_test"
    assert record["wealth_gini"] == 0.1
    assert record["num_work"] == 4
    assert record["num_save"] == 1
