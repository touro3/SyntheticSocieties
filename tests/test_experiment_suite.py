import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment_suite import build_experiment_config


def test_build_experiment_config():
    base_config = {
        "project": {"name": "bgf", "experiment_id": "base", "seed": 1},
        "policy": {"type": "mock"},
        "simulation": {"population_size": 5, "rounds": 3},
        "environment": {"public_signal": {"economy": "stable"}, "prices": {}, "resources": {}},
        "agent_defaults": {"memory_size": 10},
        "network": {"type": "random", "edge_prob": 0.5},
    }

    config = build_experiment_config(
        base_config=base_config,
        policy_type="rule_based",
        seed=42,
        experiment_id="rule_based_seed_42",
    )

    assert config["project"]["experiment_id"] == "rule_based_seed_42"
    assert config["project"]["seed"] == 42
    assert config["policy"]["type"] == "rule_based"