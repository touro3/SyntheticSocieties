"""Tests for BGF configuration schema validation."""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from configs.schema import BGFConfig, LLMConfig, PolicyConfig, NetworkConfig


class TestBaseConfigValidates:
    def test_base_config_loads_and_validates(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "base_config.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        config = BGFConfig(**raw)
        assert config.policy.type == "mock"
        assert config.simulation.rounds == 3

    def test_defaults_produce_valid_config(self):
        config = BGFConfig()
        assert config.policy.type == "mock"


class TestPolicyValidation:
    def test_unknown_policy_type_rejected(self):
        with pytest.raises(ValidationError):
            PolicyConfig(type="neural_network")

    @pytest.mark.parametrize("t", ["mock", "random", "template", "rule_based", "llm"])
    def test_valid_policy_types(self, t):
        assert PolicyConfig(type=t).type == t


class TestLLMConfig:
    def test_cache_dir_resolves_tilde(self):
        cfg = LLMConfig(cache_dir="~/models")
        assert "~" not in cfg.cache_dir
        assert cfg.cache_dir.endswith("/models")

    def test_default_cache_dir_is_none(self):
        cfg = LLMConfig()
        assert cfg.cache_dir is None

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=5.0)


class TestNetworkConfig:
    def test_invalid_network_type_rejected(self):
        with pytest.raises(ValidationError):
            NetworkConfig(type="fully_connected")

    def test_edge_prob_bounds(self):
        with pytest.raises(ValidationError):
            NetworkConfig(edge_prob=1.5)
