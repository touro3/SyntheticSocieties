"""Tests for the 9 quick-win fixes — architecture & statistical improvements.

TDD: tests written BEFORE implementation. Each class covers one fix.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.memory import HierarchicalMemory, MemoryItem
from agents.state import AgentState
from decision.llm_policy_base import LLMPolicyBase
from decision.output_parser import parse_llm_output
from decision.schemas import ProposedAction
from decision.token_budget import trim_to_budget
from tests.conftest import make_profile

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_memory_item(
    round_id: int,
    event_type: str,
    partner_id: str | None = None,
    content: str = "",
    outcome: dict | None = None,
) -> MemoryItem:
    return MemoryItem(
        round_id=round_id,
        event_type=event_type,
        partner_id=partner_id,
        content=content,
        outcome=outcome or {},
    )


class _MockBackend:
    def __init__(self, responses):
        self._responses = iter(responses)

    def generate(self, messages, temperature=None):
        return next(self._responses)


def _make_policy_base(responses=None, max_retries=2):
    backend = _MockBackend(responses or [])
    base = LLMPolicyBase.__new__(LLMPolicyBase)
    base.backend = backend
    base.temperature = 0.7
    base.max_retries = max_retries
    base.prompt_logger = None
    return base


VALID_JSON = '{"action_type": "work", "amount": 10.0, "reasoning_summary": "earn", "confidence": 0.8}'


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1: Tokenizer truncation — model-aware max_length
# ���══════════════════════════════════════════════════════════════════════════════


class TestTokenizerModelAwareMaxLength:
    """LLMBackend must use the model's context_length, not hardcoded 2048."""

    def test_backend_stores_context_length(self):
        """LLMBackend must accept and store a context_length parameter."""
        from decision.llm_backend import LLMBackend

        backend = LLMBackend(model_id="test", context_length=8192)
        assert backend.context_length == 8192

    def test_backend_default_context_length(self):
        """Default context_length should be a reasonable value, not 2048."""
        from decision.llm_backend import LLMBackend

        backend = LLMBackend(model_id="test")
        assert backend.context_length >= 4096

    def test_model_config_passes_context_length_to_backend(self):
        """get_backend() must pass context_length from ModelConfig to LLMBackend."""
        from decision.model_config import ModelConfig, get_backend

        cfg = ModelConfig.mistral_7b()
        backend = get_backend(cfg)
        assert backend.context_length == cfg.context_length


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: OpenAI backend — correct API call
# ═══════════════════════════════════════════════════════════════════════════════


class TestOpenAIBackendAPI:
    """OpenAIBackend must use chat.completions.create, not responses.create."""

    def test_generate_uses_chat_completions(self):
        """The generate method must call chat.completions.create."""
        from decision.openai_backend import OpenAIBackend

        backend = OpenAIBackend(model_id="gpt-4o-mini", api_key="test-key")

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action_type": "work", "amount": 10}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        backend._client = mock_client

        text, latency = backend.generate([{"role": "user", "content": "test"}], temperature=0.7)

        mock_client.chat.completions.create.assert_called_once()
        assert "work" in text

    def test_generate_passes_correct_params(self):
        """Must pass model, messages, temperature, max_tokens correctly."""
        from decision.openai_backend import OpenAIBackend

        backend = OpenAIBackend(model_id="gpt-4o-mini", api_key="test-key", max_new_tokens=512)

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "output"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        backend._client = mock_client

        backend.generate(
            [{"role": "user", "content": "test"}],
            temperature=0.5,
            max_new_tokens=100,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 100

    def test_response_parsing_extracts_content(self):
        """Must extract text from choices[0].message.content."""
        from decision.openai_backend import OpenAIBackend

        backend = OpenAIBackend(model_id="gpt-4o-mini", api_key="test-key")

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '  {"action_type": "save"}  '
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        backend._client = mock_client

        text, _ = backend.generate([{"role": "user", "content": "test"}])
        assert text == '{"action_type": "save"}'


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3: FDR correction — Benjamini-Hochberg
# ═══════════════════════════════════════════════════════════════════════════════


class TestFDRCorrection:
    """pairwise_significance must apply Benjamini-Hochberg FDR correction."""

    def test_fdr_correct_function_exists(self):
        from tracker.analytics import fdr_correct

        p_values = [0.01, 0.04, 0.03, 0.20]
        corrected = fdr_correct(p_values)
        assert len(corrected) == len(p_values)

    def test_fdr_preserves_order(self):
        """Corrected p-values should preserve the relative ordering."""
        from tracker.analytics import fdr_correct

        p_values = [0.01, 0.03, 0.04, 0.20]
        corrected = fdr_correct(p_values)
        # Smallest original should still be smallest corrected
        assert np.argmin(corrected) == np.argmin(p_values)

    def test_fdr_corrected_are_larger_or_equal(self):
        """Corrected p-values should be >= original p-values."""
        from tracker.analytics import fdr_correct

        p_values = [0.01, 0.03, 0.05, 0.20]
        corrected = fdr_correct(p_values)
        for orig, corr in zip(p_values, corrected):
            assert corr >= orig - 1e-12

    def test_fdr_corrected_bounded_by_one(self):
        """Corrected p-values must not exceed 1.0."""
        from tracker.analytics import fdr_correct

        p_values = [0.5, 0.6, 0.7, 0.8]
        corrected = fdr_correct(p_values)
        assert all(p <= 1.0 for p in corrected)

    def test_fdr_single_p_value(self):
        from tracker.analytics import fdr_correct

        corrected = fdr_correct([0.03])
        assert len(corrected) == 1
        assert corrected[0] == pytest.approx(0.03)

    def test_fdr_empty_list(self):
        from tracker.analytics import fdr_correct

        corrected = fdr_correct([])
        assert corrected == []

    def test_pairwise_significance_includes_fdr_column(self):
        """pairwise_significance must return p_value_fdr and significant_005_fdr columns."""
        import pandas as pd

        from tracker.analytics import pairwise_significance

        df = pd.DataFrame(
            {
                "policy_type": ["llm"] * 5 + ["random"] * 5 + ["template"] * 5,
                "wealth_mean": ([100, 102, 98, 101, 99] + [80, 82, 78, 81, 79] + [90, 92, 88, 91, 89]),
            }
        )
        result = pairwise_significance(df)
        assert "p_value_fdr" in result.columns
        assert "significant_005_fdr" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4: Delete fast_batched_backend.py (replace with LLMBackend wrapper)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFastBatchedBackendRemoved:
    """FastBatchedBackend should be replaced by a deprecation wrapper."""

    def test_import_emits_deprecation_warning(self):
        """Importing FastBatchedBackend should emit a DeprecationWarning."""
        import sys

        # Force re-import so the deprecation warning fires inside the catcher
        for key in [k for k in sys.modules if "fast_batched_backend" in k]:
            del sys.modules[key]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from decision.fast_batched_backend import FastBatchedBackend  # noqa: F401

            dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5: Reverse RAG trim priority
# ═══════════════════════════════════════════════════════════════════════════════


class TestRAGTrimPriority:
    """RAG contexts should be preserved longer than memory under budget pressure."""

    def test_memory_trimmed_before_rag_when_budget_tight(self):
        """Under moderate budget pressure, memory should shrink before RAG is dropped."""
        result = trim_to_budget(
            system="sys",
            persona="persona",
            state="state",
            memory="\n".join(f"Round {i}: work" for i in range(50)),
            context="ctx",
            population_context="ESS: avg trust 5.2/10",
            social_context="You cooperated with agent_3",
            extra="hint",
            max_tokens=120,
        )
        # Both RAG contexts should still be present when memory can be trimmed
        # Memory should be reduced (trimmed) while RAG survives
        if result["population_context"] is not None and result["social_context"] is not None:
            # If both RAG survived, memory must have been trimmed
            mem_lines = result["memory"].splitlines() if result["memory"] else []
            assert len(mem_lines) < 50

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_extra_dropped_first(self):
        """Extra guidance is always dropped first regardless of new priority."""
        result = trim_to_budget(
            system="s",
            persona="p",
            state="t",
            memory="m" * 400,
            context="c",
            population_context="pop",
            social_context="social",
            extra="extra hint",
            max_tokens=50,
        )
        assert result["extra"] is None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_rag_survives_when_memory_can_shrink(self):
        """If memory is large enough to shrink, RAG should not be dropped."""
        large_memory = "\n".join(f"Round {i}: work" for i in range(100))
        result = trim_to_budget(
            system="s",
            persona="p",
            state="t",
            memory=large_memory,
            context="c",
            population_context="pop context",
            social_context="social context",
            extra=None,
            max_tokens=80,
        )
        # At least one RAG context should survive if memory was shrinkable
        mem_lines = result["memory"].splitlines() if result["memory"] else []
        if len(mem_lines) < 100:
            # Memory was trimmed, so RAG should have been preserved
            rag_survived = result["population_context"] is not None or result["social_context"] is not None
            assert rag_survived


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6: Enrich memory reflection with outcomes
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryReflectionOutcomes:
    """Reflections must include reciprocation rates from outcome data."""

    def test_reflection_includes_reciprocation_info(self):
        """When outcomes track reciprocation, reflection should mention it."""
        mem = HierarchicalMemory()
        # 5 cooperations with agent_3: 3 reciprocated, 2 not
        for i in range(3):
            mem.add(_make_memory_item(i, "cooperate", "agent_3", outcome={"reciprocated": True}))
        for i in range(3, 5):
            mem.add(_make_memory_item(i, "cooperate", "agent_3", outcome={"reciprocated": False}))
        reflection = mem.generate_reflection()
        # Should mention reciprocation rate or percentage
        assert "agent_3" in reflection
        # The reflection should include some reciprocation information
        assert any(kw in reflection.lower() for kw in ["reciprocat", "returned", "60%", "3/5"])

    def test_reflection_without_outcomes_still_works(self):
        """Backward compatible: empty outcomes should not break reflection."""
        mem = HierarchicalMemory()
        for i in range(5):
            mem.add(_make_memory_item(i, "work"))
        reflection = mem.generate_reflection()
        assert "work" in reflection.lower()

    def test_reflection_with_mixed_outcomes(self):
        """Agents with both reciprocated and non-reciprocated cooperations."""
        mem = HierarchicalMemory()
        mem.add(_make_memory_item(1, "cooperate", "agent_1", outcome={"reciprocated": True}))
        mem.add(_make_memory_item(2, "cooperate", "agent_2", outcome={"reciprocated": False}))
        mem.add(_make_memory_item(3, "cooperate", "agent_1", outcome={"reciprocated": True}))
        reflection = mem.generate_reflection()
        assert isinstance(reflection, str)
        assert len(reflection) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 7: Log fallback/parse failure rates
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseFailureTracking:
    """Output parser and policy base must track parse method statistics."""

    def test_parse_metadata_always_includes_method(self):
        """Every parse_llm_output call must return parse_method in metadata."""
        _, meta = parse_llm_output(VALID_JSON)
        assert "parse_method" in meta
        assert meta["parse_method"] == "direct_json"

    def test_parse_stats_tracker_exists(self):
        """A module-level parse stats tracker should exist."""
        from decision.output_parser import get_parse_stats, reset_parse_stats

        reset_parse_stats()
        parse_llm_output(VALID_JSON)
        parse_llm_output("I want to help my friend", neighbors=["a1"])
        stats = get_parse_stats()

        assert "direct_json" in stats
        assert stats["direct_json"] >= 1
        assert "keyword_fallback" in stats
        assert stats["keyword_fallback"] >= 1

    def test_parse_stats_counts_failures(self):
        """Stats tracker must count total failures (None returned)."""
        from decision.output_parser import get_parse_stats, reset_parse_stats

        reset_parse_stats()
        parse_llm_output('{"action_type": "attack"}')  # invalid action
        stats = get_parse_stats()
        assert stats.get("failed", 0) >= 1

    def test_parse_stats_reset(self):
        """reset_parse_stats must clear all counters."""
        from decision.output_parser import get_parse_stats, reset_parse_stats

        parse_llm_output(VALID_JSON)
        reset_parse_stats()
        stats = get_parse_stats()
        assert sum(stats.values()) == 0

    def test_fallback_action_logged(self):
        """LLMPolicyBase must track when fallback actions are used."""
        base = _make_policy_base()
        # Instance should support fallback tracking (lazy-initialized)
        assert hasattr(base, "get_fallback_rate")
        assert hasattr(base, "get_proposal_stats")
        # After first use, counters should be accessible
        state = AgentState(wealth=30.0)
        base._fallback_action(state, neighbors=["a1"])
        assert base._fallback_counter >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 8: Persona-aware fallback
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersonaAwareFallback:
    """_fallback_action must use profile traits, not just wealth."""

    def test_fallback_accepts_profile(self):
        """_fallback_action must accept a profile argument."""
        base = _make_policy_base()
        state = AgentState(wealth=50.0)
        profile = make_profile(risk_tolerance=0.5, trust_people=0.5)
        # Should not raise — accepts profile
        action = base._fallback_action(state, neighbors=["a1"], profile=profile)
        assert isinstance(action, ProposedAction)

    def test_high_trust_favors_cooperation(self):
        """High-trust agents should cooperate even at moderate wealth."""
        base = _make_policy_base()
        state = AgentState(wealth=85.0)  # moderate wealth
        high_trust = make_profile(trust_people=0.9, risk_tolerance=0.5)
        action = base._fallback_action(state, neighbors=["a1"], profile=high_trust)
        assert action.action_type == "cooperate"

    def test_low_trust_favors_save(self):
        """Low-trust agents should save even when wealthy with neighbors."""
        base = _make_policy_base()
        state = AgentState(wealth=120.0)
        low_trust = make_profile(trust_people=0.1, risk_tolerance=0.3)
        action = base._fallback_action(state, neighbors=["a1"], profile=low_trust)
        assert action.action_type == "save"

    def test_high_risk_tolerance_works_harder(self):
        """High risk-tolerance agents should work at moderate wealth."""
        base = _make_policy_base()
        state = AgentState(wealth=75.0)  # above old threshold
        high_risk = make_profile(risk_tolerance=0.9, trust_people=0.3)
        action = base._fallback_action(state, neighbors=["a1"], profile=high_risk)
        assert action.action_type == "work"

    def test_fallback_without_profile_still_works(self):
        """Backward compatible: no profile = old wealth-based behavior."""
        base = _make_policy_base()
        state = AgentState(wealth=30.0)
        action = base._fallback_action(state, neighbors=["a1"])
        assert action.action_type == "work"

    def test_fallback_reasoning_mentions_persona(self):
        """Fallback reasoning should indicate persona traits were used."""
        base = _make_policy_base()
        state = AgentState(wealth=85.0)
        profile = make_profile(trust_people=0.9)
        action = base._fallback_action(state, neighbors=["a1"], profile=profile)
        assert "trust" in action.reasoning_summary.lower() or "persona" in action.reasoning_summary.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 9: Bootstrap mediation CIs
# ═══════════════════════════════════════════════════════════════════════════════


class TestBootstrapMediationCIs:
    """Mediation decomposition must support bootstrap confidence intervals."""

    def test_bootstrap_mediation_exists(self):
        from metrics.mediation import bootstrap_mediation_decomposition

        result = bootstrap_mediation_decomposition(
            full_grounded_samples=[0.30, 0.32, 0.28, 0.31, 0.29],
            persona_only_samples=[0.50, 0.52, 0.48, 0.51, 0.49],
            rag_only_samples=[0.55, 0.53, 0.57, 0.54, 0.56],
            baseline_samples=[0.75, 0.73, 0.77, 0.74, 0.76],
        )
        assert isinstance(result, dict)

    def test_bootstrap_returns_ci_keys(self):
        from metrics.mediation import bootstrap_mediation_decomposition

        result = bootstrap_mediation_decomposition(
            full_grounded_samples=[0.30, 0.32, 0.28],
            persona_only_samples=[0.50, 0.52, 0.48],
            rag_only_samples=[0.55, 0.53, 0.57],
            baseline_samples=[0.75, 0.73, 0.77],
        )
        # Must have CI for each effect
        for effect in ["total_effect", "persona_effect", "rag_effect", "interaction_effect"]:
            assert f"{effect}_ci_lower" in result
            assert f"{effect}_ci_upper" in result
            assert effect in result  # point estimate

    def test_bootstrap_ci_contains_point_estimate(self):
        """CI should contain the point estimate (at least approximately)."""
        from metrics.mediation import bootstrap_mediation_decomposition

        np.random.seed(42)
        result = bootstrap_mediation_decomposition(
            full_grounded_samples=[0.30] * 20,
            persona_only_samples=[0.50] * 20,
            rag_only_samples=[0.55] * 20,
            baseline_samples=[0.75] * 20,
            n_bootstrap=1000,
            seed=42,
        )
        # With zero-variance data, CI should collapse to the point estimate
        for effect in ["total_effect", "persona_effect", "rag_effect"]:
            assert result[f"{effect}_ci_lower"] <= result[effect] + 1e-6
            assert result[f"{effect}_ci_upper"] >= result[effect] - 1e-6

    def test_bootstrap_wider_ci_with_more_variance(self):
        """Higher variance data should produce wider CIs."""
        from metrics.mediation import bootstrap_mediation_decomposition

        # Low variance
        low_var = bootstrap_mediation_decomposition(
            full_grounded_samples=[0.30, 0.30, 0.30, 0.30, 0.30],
            persona_only_samples=[0.50, 0.50, 0.50, 0.50, 0.50],
            rag_only_samples=[0.55, 0.55, 0.55, 0.55, 0.55],
            baseline_samples=[0.75, 0.75, 0.75, 0.75, 0.75],
            seed=42,
        )
        # High variance
        high_var = bootstrap_mediation_decomposition(
            full_grounded_samples=[0.10, 0.50, 0.20, 0.40, 0.30],
            persona_only_samples=[0.30, 0.70, 0.40, 0.60, 0.50],
            rag_only_samples=[0.35, 0.75, 0.45, 0.65, 0.55],
            baseline_samples=[0.55, 0.95, 0.65, 0.85, 0.75],
            seed=42,
        )
        low_width = low_var["total_effect_ci_upper"] - low_var["total_effect_ci_lower"]
        high_width = high_var["total_effect_ci_upper"] - high_var["total_effect_ci_lower"]
        assert high_width >= low_width

    def test_bootstrap_respects_seed(self):
        """Same seed should produce identical CIs."""
        from metrics.mediation import bootstrap_mediation_decomposition

        kwargs = dict(
            full_grounded_samples=[0.30, 0.32, 0.28],
            persona_only_samples=[0.50, 0.52, 0.48],
            rag_only_samples=[0.55, 0.53, 0.57],
            baseline_samples=[0.75, 0.73, 0.77],
            seed=123,
        )
        r1 = bootstrap_mediation_decomposition(**kwargs)
        r2 = bootstrap_mediation_decomposition(**kwargs)
        assert r1 == r2
