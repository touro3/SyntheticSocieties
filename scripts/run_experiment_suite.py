import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import load_config
from utils.io import ensure_dir, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BGF experiment suite.")
    parser.add_argument(
        "--suite-config",
        type=str,
        default="configs/experiment_suite.yaml",
        help="Path to suite YAML config file.",
    )
    return parser.parse_args()


def build_experiment_config(base_config: dict, experiment_spec: dict) -> dict:
    config = dict(base_config)

    config["project"] = dict(base_config["project"])
    config["policy"] = dict(base_config["policy"])
    config["simulation"] = dict(base_config["simulation"])
    config["environment"] = dict(base_config["environment"])
    config["agent_defaults"] = dict(base_config["agent_defaults"])
    config["network"] = dict(base_config["network"])

    config["project"]["experiment_id"] = experiment_spec["experiment_id"]
    config["project"]["seed"] = experiment_spec["seed"]
    config["policy"]["type"] = experiment_spec["policy"]
    config["network"]["type"] = experiment_spec["network"]

    if "k" in experiment_spec:
        config["network"]["k"] = experiment_spec["k"]

    if "rewiring_prob" in experiment_spec:
        config["network"]["rewiring_prob"] = experiment_spec["rewiring_prob"]

    if "edge_prob" in experiment_spec:
        config["network"]["edge_prob"] = experiment_spec["edge_prob"]

    if "population_size" in experiment_spec:
        config["simulation"]["population_size"] = experiment_spec["population_size"]

    if "rounds" in experiment_spec:
        config["simulation"]["rounds"] = experiment_spec["rounds"]

    return config


def main() -> None:
    args = parse_args()

    suite_config = load_config(args.suite_config)
    base_config = load_config(suite_config["base_config_path"])

    generated_dir = ensure_dir("configs/generated")
    experiments = suite_config["experiments"]

    for experiment_spec in experiments:
        experiment_id = experiment_spec["experiment_id"]

        config = build_experiment_config(
            base_config=base_config,
            experiment_spec=experiment_spec,
        )

        config_path = generated_dir / f"{experiment_id}.yaml"
        save_yaml(config, config_path)

        print(f"Running experiment: {experiment_id}")

        subprocess.run(
            [
                sys.executable,
                "scripts/run_config_simulation.py",
                "--config",
                str(config_path),
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/register_experiment.py",
                "--run-dir",
                f"experiments/{experiment_id}",
            ],
            check=True,
        )

    print("Experiment suite completed.")
    print("Registered all runs into tracker.")


if __name__ == "__main__":
    main()