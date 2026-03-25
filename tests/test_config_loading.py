from utils.config import load_config


def test_load_config():
    config = load_config("configs/base_config.yaml")

    assert config["project"]["name"] == "bgf"
    assert config["simulation"]["rounds"] == 3
    assert config["simulation"]["population_size"] == 5
