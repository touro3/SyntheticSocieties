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


def build_experiment_config(base_config: dict, policy_type: str, seed: int, experiment_id: str) -> dict:
    config = dict(base_config)

    config["project"] = dict(base_config["project"])
    config["policy"] = dict(base_config["policy"])
    config["simulation"] = dict(base_config["simulation"])
    config["environment"] = dict(base_config["environment"])
    config["agent_defaults"] = dict(base_config["agent_defaults"])
    config["network"] = dict(base_config["network"])

    config["project"]["experiment_id"] = experiment_id
    config["project"]["seed"] = seed
    config["policy"]["type"] = policy_type

    return config


def main() -> None:
    args = parse_args()

    suite_config = load_config(args.suite_config)
    base_config = load_config(suite_config["base_config_path"])

    generated_dir = ensure_dir("configs/generated")

    policy_types = suite_config["sweep"]["policy_types"]
    seeds = suite_config["sweep"]["seeds"]

    for policy_type in policy_types:
        for seed in seeds:
            experiment_id = f"{policy_type}_seed_{seed}"

            config = build_experiment_config(
                base_config=base_config,
                policy_type=policy_type,
                seed=seed,
                experiment_id=experiment_id,
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