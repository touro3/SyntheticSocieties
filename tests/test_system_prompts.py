"""Tests for centralized system prompts registry."""

import pytest

from decision.system_prompts import (
    NEUTRAL_SYSTEM_PROMPT,
    PERSONA_LOCKED_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    get_system_prompt,
)


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


# ── Phase 4: alignment-bias / identity-leakage audit ─────────────────────────

# Lexicon that would indicate RLHF/assistant-helpfulness leakage.
_HELPFULNESS_LEXICON = (
    "helpful assistant",
    "as an ai",
    "as an ai language model",
    "i am an ai",
    "language model",
    "i'm here to help",
    "be helpful",
    "harmless",
)


class TestPersonaLockedAndAlignmentAudit:
    def test_persona_locked_registered(self):
        assert "persona_locked" in SYSTEM_PROMPTS
        assert get_system_prompt("persona_locked") == PERSONA_LOCKED_SYSTEM_PROMPT

    @pytest.mark.parametrize("mode", list(SYSTEM_PROMPTS.keys()))
    def test_no_prompt_leaks_helpfulness_lexicon(self, mode):
        low = SYSTEM_PROMPTS[mode].lower()
        for term in _HELPFULNESS_LEXICON:
            assert term not in low, f"prompt {mode!r} leaks alignment lexicon: {term!r}"

    def test_persona_locked_forbids_ai_framing(self):
        low = PERSONA_LOCKED_SYSTEM_PROMPT.lower()
        assert "not an ai" in low
        assert "do not break character" in low

    def test_persona_locked_keeps_neutral_mechanics(self):
        for line in (
            '- "work": Earn income.',
            '- "save": Rest and preserve wealth.',
            '- "cooperate": Share resources with a neighbor.',
        ):
            assert line in NEUTRAL_SYSTEM_PROMPT
            assert line in PERSONA_LOCKED_SYSTEM_PROMPT

    def test_existing_aliases_unchanged_regression(self):
        assert SYSTEM_PROMPTS["base"] == SYSTEM_PROMPTS["neutral"]
        assert SYSTEM_PROMPTS["balanced"] == SYSTEM_PROMPTS["neutral"]
        assert NEUTRAL_SYSTEM_PROMPT.startswith("You are a person living in a simulated society.")
