import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import load_config


def test_load_config():
    config = load_config("configs/base_config.yaml")

    assert config["project"]["name"] == "bgf"
    assert config["simulation"]["rounds"] == 3
    assert config["simulation"]["population_size"] == 5
