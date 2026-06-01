from utils.config import load_bgf_config, load_config


def test_load_config_returns_dict():
    """load_config() returns a raw dict for generic YAML files."""
    config = load_config("configs/base_config.yaml")
    assert isinstance(config, dict)
    assert config["project"]["name"] == "bgf"


def test_load_bgf_config_returns_validated_object():
    """load_bgf_config() returns a validated BGFConfig instance."""
    from configs.schema import BGFConfig

    config = load_bgf_config("configs/base_config.yaml")
    assert isinstance(config, BGFConfig)


def test_load_bgf_config_attribute_access():
    config = load_bgf_config("configs/base_config.yaml")
    assert config.project.name == "bgf"
    assert config.simulation.rounds == 30
    assert config.simulation.population_size == 100
    assert config.policy.type == "llm"
    assert config.llm.inference_timeout == 120


def test_load_bgf_config_rejects_bad_value(tmp_path):
    """A config with an invalid field value raises ValidationError at load time."""
    from pydantic import ValidationError

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("simulation:\n  rounds: -1\n")
    try:
        load_bgf_config(bad_yaml)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # expected
