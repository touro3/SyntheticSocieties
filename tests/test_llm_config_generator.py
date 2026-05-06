"""Tests for LLMConfigGenerator and utility functions (configs/llm_config_generator.py)."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from configs.llm_config_generator import (
    LLMConfigGenerator,
    _deep_merge,
    _repair_and_parse,
)


def test_deep_merge_override_wins():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}}


def test_deep_merge_new_keys_added():
    result = _deep_merge({"a": 1}, {"b": 2})
    assert result == {"a": 1, "b": 2}


def test_repair_and_parse_strips_markdown_fences():
    raw = '```json\n{"key": "value"}\n```'
    result = _repair_and_parse(raw)
    assert result == {"key": "value"}


def test_repair_and_parse_plain_json():
    result = _repair_and_parse('{"x": 42}')
    assert result == {"x": 42}


def test_repair_and_parse_trailing_comma():
    result = _repair_and_parse('{"a": 1,}')
    assert result == {"a": 1}


def test_validate_clamps_population_size():
    gen = _make_generator_no_client()
    config = {"simulation": {"population_size": 9999, "rounds": 30}}
    result = gen._validate(config)
    assert result["simulation"]["population_size"] == 500


def test_validate_clamps_rounds():
    gen = _make_generator_no_client()
    config = {"simulation": {"population_size": 50, "rounds": 999}}
    result = gen._validate(config)
    assert result["simulation"]["rounds"] == 200


def test_validate_invalid_policy_defaults_to_llm():
    gen = _make_generator_no_client()
    config = {"policy": {"type": "invalid_type"}}
    result = gen._validate(config)
    assert result["policy"]["type"] == "llm"


def test_validate_known_policy_preserved():
    gen = _make_generator_no_client()
    for policy in ("random", "rule_based", "mock", "generative_agents"):
        config = {"policy": {"type": policy}}
        result = gen._validate(config)
        assert result["policy"]["type"] == policy


def test_validate_network_type_defaults_on_invalid():
    gen = _make_generator_no_client()
    config = {"network": {"type": "star", "edge_prob": 0.5, "k": 2, "rewiring_prob": 0.1}}
    result = gen._validate(config)
    assert result["network"]["type"] == "random"


def test_validate_risk_tolerance_clamped():
    gen = _make_generator_no_client()
    config = {"agent_defaults": {"risk_tolerance": 5.0, "initial_wealth": 50.0}}
    result = gen._validate(config)
    assert result["agent_defaults"]["risk_tolerance"] == 1.0


def test_generate_uses_defaults_on_all_stage_failures():
    """If every LLM call fails, generate() returns a valid dict with required keys."""
    with patch("configs.llm_config_generator.LLMConfigGenerator._call_llm", side_effect=RuntimeError("no API")):
        with patch("openai.OpenAI"):
            gen = LLMConfigGenerator(api_key="test")
            config = gen.generate("A test scenario")

    required_keys = {"project", "simulation", "policy", "population", "network", "environment", "llm"}
    assert required_keys.issubset(config.keys())
    assert isinstance(config["simulation"]["population_size"], int)
    assert isinstance(config["simulation"]["rounds"], int)


def test_save_writes_yaml(tmp_path):
    gen = _make_generator_no_client()
    config = {"project": {"name": "test"}, "simulation": {"rounds": 10, "population_size": 5}}
    out_path = tmp_path / "config.yaml"
    gen.save(config, out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert "test" in content


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_generator_no_client():
    """Create a generator instance without initialising the OpenAI client."""
    with patch("openai.OpenAI"):
        return LLMConfigGenerator(api_key="dummy")
