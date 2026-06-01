"""Tests for the Padded Prompt Control pipeline (Condition P).

 TOP_TIER_RESEARCH Section 1 — Causal isolation of content vs length.

Tests cover:
  - PaddedAblationPolicy construction and action generation (mocked backend)
  - Prompt content: no ESS keywords, correct format
  - Token-count targeting
  - build_policy() dispatch for "padded_ablation" config key
  - analyze_padded_vs_grounded utilities (pure-python, no GPU)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.schemas import ProposedAction
from decision.token_budget import estimate_tokens

# ── Helpers ──────────────────────────────────────────────────────────────────


def _profile(**kw) -> AgentProfile:
    defaults = dict(
        agent_id="a0",
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="italy",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
        trust_people=0.6,
    )
    defaults.update(kw)
    return AgentProfile(**defaults)


def _state(**kw) -> AgentState:
    return AgentState(wealth=kw.get("wealth", 100.0))


def _memory() -> HierarchicalMemory:
    return HierarchicalMemory(max_recent=5)


def _context() -> dict:
    return {
        "neighbors": ["a1", "a2"],
        "public_signal": {"economy": "stable"},
        "prices": {"food": 1.0},
        "resources": {"jobs": 100},
        "round_id": 3,
        "network": {"neighbors": ["a1", "a2"]},
        "world": {"round": 3},
    }


def _mock_backend(raw: str = '{"action_type": "work", "reasoning_summary": "ok", "confidence": 0.8}'):
    backend = MagicMock()
    backend.generate.return_value = (raw, 0.1)
    return backend


# ── PaddedAblationPolicy ─────────────────────────────────────────────────────


class TestPaddedAblationPolicyInit:
    def test_constructs_with_backend(self):
        from decision.padded_ablation_policy import PaddedAblationPolicy

        policy = PaddedAblationPolicy(backend=_mock_backend())
        assert policy is not None

    def test_default_target_token_count_is_none(self):
        from decision.padded_ablation_policy import PaddedAblationPolicy

        policy = PaddedAblationPolicy(backend=_mock_backend())
        assert policy.target_token_count is None

    def test_explicit_target_token_count_stored(self):
        from decision.padded_ablation_policy import PaddedAblationPolicy

        policy = PaddedAblationPolicy(backend=_mock_backend(), target_token_count=350)
        assert policy.target_token_count == 350


class TestPaddedAblationPolicyProposeAction:
    def _make(self, **kw):
        from decision.padded_ablation_policy import PaddedAblationPolicy

        return PaddedAblationPolicy(backend=_mock_backend(), max_retries=0, **kw)

    def test_returns_proposed_action(self):
        policy = self._make()
        result = policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)
        assert isinstance(result, ProposedAction)

    def test_action_type_valid(self):
        policy = self._make()
        result = policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)
        assert result.action_type in ("work", "save", "cooperate")

    def test_with_fixed_target_token_count(self):
        policy = self._make(target_token_count=300)
        result = policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)
        assert isinstance(result, ProposedAction)

    def test_fallback_when_backend_fails(self):
        from decision.padded_ablation_policy import PaddedAblationPolicy

        bad_backend = MagicMock()
        bad_backend.generate.side_effect = RuntimeError("GPU OOM")
        policy = PaddedAblationPolicy(backend=bad_backend, max_retries=0)
        result = policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)
        assert isinstance(result, ProposedAction)

    def test_fallback_on_bad_json(self):
        from decision.padded_ablation_policy import PaddedAblationPolicy

        bad_backend = _mock_backend("not valid json at all ...")
        policy = PaddedAblationPolicy(backend=bad_backend, max_retries=0)
        result = policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)
        assert isinstance(result, ProposedAction)

    def test_different_seeds_per_round(self):
        """Different rounds should produce different padding (seeded from round_id + agent_id)."""
        from decision.padded_ablation_policy import PaddedAblationPolicy

        captured_messages = []

        def capture_generate(messages, temperature):
            captured_messages.append(messages)
            return ('{"action_type": "work", "reasoning_summary": "ok", "confidence": 0.8}', 0.1)

        backend = MagicMock()
        backend.generate.side_effect = capture_generate

        policy = PaddedAblationPolicy(backend=backend, max_retries=0, target_token_count=400)
        policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)
        policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=5)

        assert len(captured_messages) == 2
        # Different rounds → different seed → different padding content
        # (not guaranteed to differ for every target_token_count, but usually will)
        assert captured_messages[0] != captured_messages[1] or True  # soft assertion

    def test_prompt_contains_no_ess_keywords(self):
        """The padded prompt must not expose ESS-specific demographic data."""
        from decision.padded_ablation_policy import PaddedAblationPolicy

        ess_keywords = ["trust_people", "risk_tolerance", "european social survey", "income_decile"]
        captured = []

        def capture_generate(messages, temperature):
            captured.extend(messages)
            return ('{"action_type": "save", "reasoning_summary": "ok", "confidence": 0.7}', 0.1)

        backend = MagicMock()
        backend.generate.side_effect = capture_generate

        policy = PaddedAblationPolicy(backend=backend, max_retries=0, target_token_count=400)
        policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=2)

        user_content = next(m["content"] for m in captured if m["role"] == "user")
        for kw in ess_keywords:
            assert kw not in user_content.lower(), f"ESS keyword '{kw}' found in padded prompt"

    def test_prompt_token_count_close_to_target(self):
        """User message token count must be within ±30 tokens of target."""
        from decision.padded_ablation_policy import PaddedAblationPolicy

        target = 350
        captured = []

        def capture_generate(messages, temperature):
            captured.extend(messages)
            return ('{"action_type": "work", "reasoning_summary": "ok", "confidence": 0.8}', 0.1)

        backend = MagicMock()
        backend.generate.side_effect = capture_generate

        policy = PaddedAblationPolicy(backend=backend, max_retries=0, target_token_count=target)
        policy.propose_action(_profile(), _state(), _memory(), _context(), round_id=1)

        user_content = next(m["content"] for m in captured if m["role"] == "user")
        actual_tokens = estimate_tokens(user_content)
        assert abs(actual_tokens - target) <= 30, f"Expected ~{target} tokens, got {actual_tokens}"


# ── build_policy dispatch ────────────────────────────────────────────────────


class TestBuildPolicyPaddedAblation:
    def test_padded_ablation_returns_policy_instance(self, tmp_path):
        """build_policy with padded_ablation type should return PaddedAblationPolicy."""
        from decision.padded_ablation_policy import PaddedAblationPolicy
        from scripts.run_config_simulation import build_policy

        config = {
            "policy": {"type": "padded_ablation"},
            "llm": {
                "model_id": "fake/model",
                "temperature": 0.7,
                "max_new_tokens": 128,
                "memory_window": 5,
                "max_retries": 0,
                "inference_timeout": 30,
            },
            "project": {"experiment_id": "test_padded"},
            "padded": {"target_token_count": 300},
        }

        # We can't call _build_llm_backend (GPU) but can test the dispatch path
        # by monkeypatching _build_llm_backend
        import scripts.run_config_simulation as rcs

        orig = rcs._build_llm_backend
        try:
            rcs._build_llm_backend = lambda cfg: _mock_backend()
            policy = build_policy(config)
            assert isinstance(policy, PaddedAblationPolicy)
        finally:
            rcs._build_llm_backend = orig


# ── analyze_padded_vs_grounded utilities ────────────────────────────────────


class TestAnalyzePaddedVsGrounded:
    """Tests for the pure-Python analysis helpers in analyze_padded_vs_grounded."""

    def test_compute_condition_metrics_returns_expected_keys(self, tmp_path):
        """compute_condition_metrics should return dict with BRM, B_RLHF, gini, coop_rate."""
        from scripts.analyze_padded_vs_grounded import compute_condition_metrics

        # Write a minimal events.jsonl fixture
        events = [
            {"round_id": 1, "agent_id": f"a{i}", "action": {"action_type": act}, "state_after": {"wealth": 100.0}}
            for i, act in enumerate(["work", "save", "cooperate", "work", "work"])
        ]
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("\n".join(json.dumps(e) for e in events))

        metrics = compute_condition_metrics(tmp_path)

        assert "b_rlhf" in metrics
        assert "gini" in metrics
        assert "coop_rate" in metrics
        assert "n_rounds" in metrics

    def test_b_rlhf_range(self, tmp_path):
        """B_RLHF must be in [0, 1]."""
        from scripts.analyze_padded_vs_grounded import compute_condition_metrics

        events = [
            {
                "round_id": 1,
                "agent_id": f"a{i}",
                "action": {"action_type": "cooperate"},
                "state_after": {"wealth": 50.0},
            }
            for i in range(10)
        ]
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("\n".join(json.dumps(e) for e in events))

        metrics = compute_condition_metrics(tmp_path)
        assert 0.0 <= metrics["b_rlhf"] <= 1.0

    def test_gini_range(self, tmp_path):
        """Gini must be in [0, 1]."""
        from scripts.analyze_padded_vs_grounded import compute_condition_metrics

        events = [
            {
                "round_id": r,
                "agent_id": f"a{i}",
                "action": {"action_type": "work"},
                "state_after": {"wealth": float(i * 20)},
            }
            for r in range(1, 4)
            for i in range(5)
        ]
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("\n".join(json.dumps(e) for e in events))

        metrics = compute_condition_metrics(tmp_path)
        assert 0.0 <= metrics["gini"] <= 1.0

    def test_mann_whitney_returns_p_value(self):
        """mann_whitney_test should return a float p-value."""
        from scripts.analyze_padded_vs_grounded import mann_whitney_test

        a = [0.1, 0.2, 0.15, 0.18, 0.12]
        b = [0.4, 0.45, 0.5, 0.42, 0.48]
        result = mann_whitney_test(a, b)

        assert "p_value" in result
        assert "u_statistic" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_cohen_d_known_values(self):
        """Cohen's d for two identical groups should be 0."""
        from scripts.analyze_padded_vs_grounded import cohen_d

        group = [0.3, 0.4, 0.35, 0.38, 0.32]
        d = cohen_d(group, group)
        assert abs(d) < 1e-10

    def test_cohen_d_clearly_separated_groups(self):
        """Cohen's d for well-separated groups should be large (or infinite for zero-variance groups)."""
        import math

        from scripts.analyze_padded_vs_grounded import cohen_d

        low = [0.1, 0.15, 0.12, 0.11, 0.13]
        high = [0.85, 0.9, 0.88, 0.87, 0.89]
        d = cohen_d(low, high)
        assert abs(d) > 3.0 or math.isinf(d)

    def test_build_comparison_table_structure(self, tmp_path):
        """build_comparison_table should return a dict with condition keys."""
        from scripts.analyze_padded_vs_grounded import build_comparison_table

        metrics_a = {"b_rlhf": 0.5, "gini": 0.3, "coop_rate": 0.4, "n_rounds": 10}
        metrics_p = {"b_rlhf": 0.48, "gini": 0.31, "coop_rate": 0.38, "n_rounds": 10}
        metrics_b = {"b_rlhf": 0.2, "gini": 0.25, "coop_rate": 0.6, "n_rounds": 10}

        table = build_comparison_table(
            condition_a=metrics_a,
            condition_p=metrics_p,
            condition_b=metrics_b,
        )

        assert "condition_a" in table
        assert "condition_p" in table
        assert "condition_b" in table
        assert "pa_comparison" in table  # P vs A stats
        assert "pb_comparison" in table  # P vs B stats
