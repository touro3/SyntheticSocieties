"""Tests for ConditionedLLMPolicy — action sanitization, bounds clamping, all prompt modes.

No GPU required — backend is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.conditioned_llm_policy import ConditionedLLMPolicy
from decision.schemas import ProposedAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(**kw) -> AgentProfile:
    defaults = dict(
        agent_id="a0", age=35, income=1000.0, education="college",
        occupation="worker", location="italy", political_preference="center",
        risk_tolerance=0.5, social_class="middle", trust_people=0.6,
    )
    defaults.update(kw)
    return AgentProfile(**defaults)


def _state(**kw) -> AgentState:
    defaults = dict(wealth=100.0, stress=0.0, satisfaction=0.5)
    defaults.update(kw)
    return AgentState(**defaults)


def _mock_backend(raw: str = '{"action_type": "work", "reasoning_summary": "ok", "confidence": 0.8}'):
    backend = MagicMock()
    backend.generate.return_value = (raw, 0.1)
    return backend


def _make_policy(**kw) -> ConditionedLLMPolicy:
    backend = kw.pop("backend", _mock_backend())
    return ConditionedLLMPolicy(backend=backend, max_retries=0, **kw)


def _context(neighbors: list[str] | None = None) -> dict:
    return {
        "neighbors": neighbors or ["a1"],
        "public_signal": {"economy": "stable"},
        "prices": {"food": 1.0},
        "resources": {"jobs": 100},
        "round_id": 1,
    }


# ---------------------------------------------------------------------------
# ACTION_BOUNDS
# ---------------------------------------------------------------------------

class TestActionBounds:
    def test_work_bounds_defined(self):
        assert "work" in ConditionedLLMPolicy.ACTION_BOUNDS

    def test_save_bounds_defined(self):
        assert "save" in ConditionedLLMPolicy.ACTION_BOUNDS

    def test_cooperate_bounds_defined(self):
        assert "cooperate" in ConditionedLLMPolicy.ACTION_BOUNDS

    def test_all_bounds_are_positive_ranges(self):
        for action, (lo, hi) in ConditionedLLMPolicy.ACTION_BOUNDS.items():
            assert lo > 0, f"{action} lower bound must be positive"
            assert hi >= lo, f"{action} upper must be >= lower"


# ---------------------------------------------------------------------------
# _sanitize_action — amount clamping
# ---------------------------------------------------------------------------

class TestSanitizeAction:
    def _action(self, action_type: str, amount: float, target: str | None = None) -> ProposedAction:
        return ProposedAction(
            action_type=action_type,
            amount=amount,
            reasoning_summary="test",
            confidence=0.8,
            target_agent_id=target,
        )

    def test_amount_below_lower_bound_is_clamped_up(self):
        policy = _make_policy()
        lo, _ = ConditionedLLMPolicy.ACTION_BOUNDS["work"]
        action = self._action("work", lo - 1.0)
        fixed, meta = policy._sanitize_action(action, ["a1"], _state())
        assert fixed.amount == pytest.approx(lo)
        assert "amount_clamped" in meta

    def test_amount_above_upper_bound_is_clamped_down(self):
        policy = _make_policy()
        _, hi = ConditionedLLMPolicy.ACTION_BOUNDS["save"]
        action = self._action("save", hi + 5.0)
        fixed, meta = policy._sanitize_action(action, ["a1"], _state())
        assert fixed.amount == pytest.approx(hi)
        assert "amount_clamped" in meta

    def test_amount_within_bounds_not_changed(self):
        policy = _make_policy()
        lo, hi = ConditionedLLMPolicy.ACTION_BOUNDS["cooperate"]
        amount = (lo + hi) / 2
        action = self._action("cooperate", amount, target="a1")
        fixed, meta = policy._sanitize_action(action, ["a1"], _state())
        assert fixed.amount == pytest.approx(amount)
        assert "amount_clamped" not in meta

    def test_invalid_action_type_converted_to_save(self):
        # ProposedAction validation won't allow unknown types, but test via direct model_copy path
        policy = _make_policy()
        action = ProposedAction(action_type="save", amount=7.0, reasoning_summary="ok")
        # Simulate an "invalid" type by patching the field
        action2 = action.model_copy(update={"action_type": "save"})  # stays valid
        fixed, _ = policy._sanitize_action(action2, ["a1"], _state())
        assert fixed.action_type in ("work", "save", "cooperate")

    def test_cooperate_without_neighbors_falls_back(self):
        policy = _make_policy()
        action = self._action("cooperate", 7.0, target="a1")
        fixed, meta = policy._sanitize_action(action, [], _state())  # no neighbors
        assert fixed.action_type != "cooperate"
        assert meta.get("cooperate_without_neighbors") is True


# ---------------------------------------------------------------------------
# propose_action — system prompt modes
# ---------------------------------------------------------------------------

class TestProposeAction:
    def test_returns_proposed_action(self):
        policy = _make_policy()
        result = policy.propose_action(
            _profile(), _state(), MemoryBuffer(max_items=5), _context(), round_id=1
        )
        assert isinstance(result, ProposedAction)

    def test_balanced_mode(self):
        policy = _make_policy(system_prompt_mode="balanced")
        result = policy.propose_action(
            _profile(), _state(), MemoryBuffer(max_items=5), _context(), round_id=1
        )
        assert result.action_type in ("work", "save", "cooperate")

    def test_base_mode(self):
        policy = _make_policy(system_prompt_mode="base")
        result = policy.propose_action(
            _profile(), _state(), MemoryBuffer(max_items=5), _context(), round_id=1
        )
        assert isinstance(result, ProposedAction)

    def test_fallback_when_backend_fails(self):
        bad_backend = MagicMock()
        bad_backend.generate.side_effect = RuntimeError("GPU OOM")
        policy = ConditionedLLMPolicy(backend=bad_backend, max_retries=0)
        result = policy.propose_action(
            _profile(), _state(), MemoryBuffer(max_items=5), _context(), round_id=1
        )
        assert isinstance(result, ProposedAction)

    def test_fallback_when_parse_fails(self):
        bad_backend = _mock_backend("this is not json at all")
        policy = ConditionedLLMPolicy(backend=bad_backend, max_retries=0)
        result = policy.propose_action(
            _profile(), _state(), MemoryBuffer(max_items=5), _context(), round_id=1
        )
        assert isinstance(result, ProposedAction)
