"""Tests for prompt token budget management."""

import pytest

from decision.token_budget import (
    DEFAULT_MAX_TOKENS,
    budget_for_model,
    estimate_tokens,
    fits_in_budget,
    trim_to_budget,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") >= 1  # minimum 1

    def test_known_length(self):
        # 400 chars at ~3.3 chars/token ≈ 121 tokens
        assert estimate_tokens("a" * 400) == 121

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
        for key in ["system", "persona", "state", "context", "memory", "population_context", "social_context", "extra"]:
            assert key in result

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_system_never_dropped(self):
        result = self._trim_tight()
        assert result["system"] is not None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_persona_never_dropped(self):
        result = self._trim_tight()
        assert result["persona"] is not None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_state_never_dropped(self):
        result = self._trim_tight()
        assert result["state"] is not None

    @pytest.mark.filterwarnings("ignore::UserWarning")
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

    def test_warning_emitted_when_social_context_dropped(self):
        with pytest.warns(UserWarning, match="dropped 'social_context'"):
            trim_to_budget(
                system="s",
                persona="p",
                state="t",
                memory="m" * 600,
                context="c",
                population_context=None,
                social_context="social" * 100,
                extra=None,
                max_tokens=50,
            )

    def test_warning_emitted_when_population_context_dropped(self):
        with pytest.warns(UserWarning, match="dropped 'population_context'"):
            trim_to_budget(
                system="s",
                persona="p",
                state="t",
                memory="m" * 600,
                context="c",
                population_context="pop" * 100,
                social_context=None,
                extra=None,
                max_tokens=50,
            )

    def test_both_rag_contexts_dropped_when_budget_exhausted(self):
        """When both RAG contexts are present and budget is tiny, both must be dropped."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = trim_to_budget(
                system="s",
                persona="p",
                state="t",
                memory="m" * 600,
                context="c",
                population_context="pop" * 100,
                social_context="social" * 100,
                extra="extra",
                max_tokens=50,
            )
        assert result["social_context"] is None
        assert result["population_context"] is None
        dropped_keys = {str(w.message) for w in caught if issubclass(w.category, UserWarning)}
        assert any("social_context" in msg for msg in dropped_keys)
        assert any("population_context" in msg for msg in dropped_keys)

    def test_memory_overflow_is_substantially_reduced(self):
        """Iterative halving must reduce memory far below single-halve output.

        With single-halve: 200 lines → 100 lines ≈ 2750 tokens (still overflows).
        With iterative halving: 200 → 100 → 50 → ... until ≤ 4-line floor ≈ 4 lines.
        We assert the result is ≤ 10 lines, demonstrating iteration happened.
        """
        large_memory = "\n".join([f"Round {i}: action_type work" for i in range(200)])
        result = trim_to_budget(
            system="s",
            persona="p",
            state="t",
            memory=large_memory,
            context="c",
            population_context=None,
            social_context=None,
            extra=None,
            max_tokens=80,
        )
        lines = result["memory"].splitlines() if result["memory"] else []
        assert len(lines) <= 10, (
            f"Iterative halving should reduce 200 lines to ≤10; got {len(lines)}. "
            "Single-halve bug would produce 100 lines."
        )


class TestBudgetForModel:
    """Tests for the per-model prompt-budget lookup table."""

    def test_default_is_3072(self):
        assert DEFAULT_MAX_TOKENS == 3072

    def test_mistral_returns_4096(self):
        assert budget_for_model("mistralai/Mistral-7B-Instruct-v0.3") == 4096

    def test_qwen_returns_6144(self):
        assert budget_for_model("Qwen/Qwen2.5-7B-Instruct") == 6144

    def test_gpt4o_mini_returns_8192(self):
        assert budget_for_model("gpt-4o-mini") == 8192

    def test_unknown_model_returns_default(self):
        assert budget_for_model("some/totally-unknown-model-xyz") == DEFAULT_MAX_TOKENS

    def test_case_insensitive(self):
        # Matching must work regardless of the casing in the model_id string.
        assert budget_for_model("MISTRAL-7B-INSTRUCT") == 4096
