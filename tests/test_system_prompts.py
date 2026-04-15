"""Tests for centralized system prompts registry."""

import pytest

from decision.system_prompts import SYSTEM_PROMPTS, get_system_prompt


class TestGetSystemPrompt:
    @pytest.mark.parametrize(
        "mode",
        [
            "base",
            "balanced",
            "experimental_base",
            "experimental_balanced",
            "no_institutions",
        ],
    )
    def test_returns_known_modes(self, mode):
        result = get_system_prompt(mode)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_raises_on_unknown_mode(self):
        with pytest.raises(KeyError, match="Unknown system prompt mode"):
            get_system_prompt("nonexistent")

    @pytest.mark.parametrize("mode", list(SYSTEM_PROMPTS.keys()))
    def test_all_prompts_contain_json_format(self, mode):
        prompt = get_system_prompt(mode)
        assert "action_type" in prompt
        assert "JSON" in prompt

    def test_base_mentions_memories(self):
        prompt = get_system_prompt("base")
        assert "memories" in prompt.lower()

    def test_no_institutions_is_permissive(self):
        prompt = get_system_prompt("no_institutions")
        assert "anything you want" in prompt.lower()
