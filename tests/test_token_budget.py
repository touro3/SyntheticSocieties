"""Tests for prompt token budget management."""

import pytest
from decision.token_budget import estimate_tokens, fits_in_budget, trim_to_budget


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") >= 1  # minimum 1

    def test_known_length(self):
        # 400 chars ≈ 100 tokens
        assert estimate_tokens("a" * 400) == 100

    def test_longer_text_has_more_tokens(self):
        assert estimate_tokens("x" * 800) > estimate_tokens("x" * 400)


class TestFitsInBudget:
    def test_short_text_fits(self):
        assert fits_in_budget("hello world", max_tokens=100)

    def test_very_long_text_does_not_fit(self):
        assert not fits_in_budget("a" * 10000, max_tokens=100)


class TestTrimToBudget:
    def _base_kwargs(self, max_tokens=200):
        return dict(
            system="You are a participant.",
            persona="You are agent_0, age 35.",
            state="wealth=50, stress=0.2",
            memory="Round 1: work\nRound 2: save",
            context="World state: stable",
            population_context="ESS context: trust 7/10",
            social_context="You cooperated with agent_3.",
            extra="HINT: balance actions.",
            max_tokens=max_tokens,
        )

    def test_returns_all_keys(self):
        result = self._trim_normal()
        for key in ["system", "persona", "state", "context", "memory",
                    "population_context", "social_context", "extra"]:
            assert key in result

    def test_system_never_dropped(self):
        result = self._trim_tight()
        assert result["system"] is not None

    def test_persona_never_dropped(self):
        result = self._trim_tight()
        assert result["persona"] is not None

    def test_state_never_dropped(self):
        result = self._trim_tight()
        assert result["state"] is not None

    def test_extra_dropped_first_when_tight(self):
        # Very tight budget forces dropping
        result = trim_to_budget(
            system="s",
            persona="p",
            state="t",
            memory="m" * 600,
            context="c",
            population_context="pop",
            social_context="social",
            extra="extra hint",
            max_tokens=50,
        )
        # extra should be dropped before population_context
        assert result["extra"] is None

    def _trim_normal(self):
        return trim_to_budget(**self._base_kwargs(max_tokens=2000))

    def _trim_tight(self):
        return trim_to_budget(**self._base_kwargs(max_tokens=30))
