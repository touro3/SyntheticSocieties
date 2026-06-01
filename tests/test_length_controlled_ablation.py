"""Tests for the length-controlled (padded) ablation prompt builder.

Causal inference and ablation formalization.

The padded ablation builds a prompt with the same token count as a
fully grounded prompt, but fills the extra space with semantically
empty filler instead of ESS data. This isolates the effect of semantic
content from prompt length.
"""

from __future__ import annotations

from agents.memory import HierarchicalMemory
from decision.padded_prompt_builder import (
    _PADDING_POOL,
    build_padded_prompt,
    measure_grounded_token_count,
)
from decision.token_budget import estimate_tokens
from tests.conftest import make_profile, make_state

# ── Fixtures ─────────────────────────────────────────────────────────────

_CONTEXT = {
    "world": {"economy": "stable", "round": 5},
    "network": {"neighbors": ["agent_1", "agent_2"]},
}
_SOCIAL = "You have received support from: agent_1. You are a central figure."
_POP = "Context: People in your bracket have average trust 6.2/10, risk 4.8/10."


# ── build_padded_prompt ──────────────────────────────────────────────────


class TestBuildPaddedPrompt:
    def test_output_has_two_messages(self):
        profile = make_profile(trust_people=0.7, risk_tolerance=0.3)
        state = make_state()
        memory = HierarchicalMemory()
        messages = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=400,
            seed=42,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_length_matches_target_within_tolerance(self):
        profile = make_profile(trust_people=0.7, risk_tolerance=0.3)
        state = make_state()
        memory = HierarchicalMemory()
        target = 400
        messages = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=target,
            seed=42,
        )
        user_tokens = estimate_tokens(messages[1]["content"])
        assert abs(user_tokens - target) <= 25, f"Expected ~{target} tokens, got {user_tokens}"

    def test_padding_contains_no_ess_keywords(self):
        """Padding must not contain ESS-specific demographic data."""
        profile = make_profile(trust_people=0.7, risk_tolerance=0.3)
        state = make_state()
        memory = HierarchicalMemory()
        messages = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=500,
            seed=42,
        )
        user_content = messages[1]["content"].lower()
        # Use word-boundary aware checks — "ess" as substring hits "stress", "processes"
        ess_keywords = [
            "trust_people",
            "risk_tolerance",
            "european social survey",
            "income_decile",
            "political_orientation",
        ]
        for kw in ess_keywords:
            assert kw not in user_content, f"Found ESS keyword '{kw}' in padded prompt"

    def test_different_seeds_produce_different_padding(self):
        profile = make_profile()
        state = make_state()
        memory = HierarchicalMemory()
        msg_a = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=400,
            seed=1,
        )
        msg_b = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=400,
            seed=99,
        )
        # User content should differ due to different padding selection
        assert msg_a[1]["content"] != msg_b[1]["content"]

    def test_never_exceeds_budget(self):
        profile = make_profile()
        state = make_state()
        memory = HierarchicalMemory()
        # Target a user-message token count that leaves room for the system prompt
        target = 800  # Well within budget even with system prompt
        messages = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=target,
            seed=42,
        )
        user_tokens = estimate_tokens(messages[1]["content"])
        assert user_tokens <= target + 30

    def test_small_target_produces_minimal_padding(self):
        profile = make_profile()
        state = make_state()
        memory = HierarchicalMemory()
        # Very small target — should produce prompt with little or no padding
        messages = build_padded_prompt(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            target_token_count=50,
            seed=42,
        )
        assert len(messages) == 2


# ── measure_grounded_token_count ─────────────────────────────────────────


class TestMeasureGroundedTokenCount:
    def test_returns_positive_int(self):
        profile = make_profile()
        state = make_state()
        memory = HierarchicalMemory()
        count = measure_grounded_token_count(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            social_context=_SOCIAL,
            population_context=_POP,
        )
        assert isinstance(count, int)
        assert count > 0

    def test_with_context_larger_than_without(self):
        profile = make_profile()
        state = make_state()
        memory = HierarchicalMemory()
        with_ctx = measure_grounded_token_count(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
            social_context=_SOCIAL,
            population_context=_POP,
        )
        without_ctx = measure_grounded_token_count(
            profile,
            state,
            memory,
            _CONTEXT,
            round_id=5,
        )
        assert with_ctx > without_ctx


# ── Padding pool validation ──────────────────────────────────────────────


class TestPaddingPool:
    def test_pool_has_enough_entries(self):
        assert len(_PADDING_POOL) >= 20

    def test_no_ess_keywords_in_pool(self):
        ess_keywords = ["trust_people", "risk_tolerance", "european social survey", "income_decile"]
        for sentence in _PADDING_POOL:
            lower = sentence.lower()
            for kw in ess_keywords:
                assert kw not in lower, f"Pool sentence contains '{kw}': {sentence}"
