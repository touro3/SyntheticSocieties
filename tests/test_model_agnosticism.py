"""Tests for env-var model agnosticism.

Guarantees base/instruct/uncensored + backend can be swapped without code
edits, while unset env preserves the exact prior behavior (backward compat).
"""

from __future__ import annotations

import pytest

from decision.model_config import ModelConfig, get_backend


def test_default_unchanged_without_env(monkeypatch):
    for v in ("BGF_MODEL_ID", "BGF_BACKEND_TYPE", "BGF_MODEL_VARIANT"):
        monkeypatch.delenv(v, raising=False)
    cfg = ModelConfig.mistral_7b()
    assert cfg.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert cfg.backend_type == "huggingface"
    assert cfg.model_variant == "instruct"


def test_env_overrides_model_and_variant(monkeypatch):
    monkeypatch.setenv("BGF_MODEL_ID", "some-org/base-llm")
    monkeypatch.setenv("BGF_MODEL_VARIANT", "base")
    monkeypatch.setenv("BGF_BACKEND_TYPE", "huggingface")
    cfg = ModelConfig.mistral_7b()  # classmethod values must be overridden
    assert cfg.model_id == "some-org/base-llm"
    assert cfg.model_variant == "base"
    assert cfg.backend_type == "huggingface"


def test_env_override_backend_to_openai(monkeypatch):
    monkeypatch.setenv("BGF_BACKEND_TYPE", "openai")
    monkeypatch.delenv("BGF_MODEL_ID", raising=False)
    cfg = ModelConfig(model_id="gpt-4o-mini")
    assert cfg.backend_type == "openai"
    backend = get_backend(cfg)
    assert backend.__class__.__name__ == "OpenAIBackend"


def test_invalid_env_values_rejected(monkeypatch):
    monkeypatch.setenv("BGF_MODEL_VARIANT", "definitely-not-valid")
    with pytest.raises(ValueError, match="BGF_MODEL_VARIANT"):
        ModelConfig(model_id="x")

    monkeypatch.delenv("BGF_MODEL_VARIANT", raising=False)
    monkeypatch.setenv("BGF_BACKEND_TYPE", "tensorflow")
    with pytest.raises(ValueError, match="BGF_BACKEND_TYPE"):
        ModelConfig(model_id="x")


def test_prompt_budget_tracks_env_model(monkeypatch):
    monkeypatch.setenv("BGF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    cfg = ModelConfig(model_id="mistralai/Mistral-7B-Instruct-v0.3")
    # Budget is re-derived from the *effective* (env) model id.
    assert cfg.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.prompt_budget > 0
