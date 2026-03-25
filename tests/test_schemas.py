"""Tests for domain value objects — ProposedAction validation."""
import pytest
from pydantic import ValidationError

from decision.schemas import ProposedAction


# ── ActionType validation ────────────────────────────────────────────────


class TestActionTypeValidation:
    def test_rejects_invalid_string(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="steal",
                reasoning_summary="I want to steal",
            )

    def test_rejects_typo(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="wrok",
                reasoning_summary="typo",
            )

    @pytest.mark.parametrize("action", ["work", "save", "cooperate"])
    def test_accepts_valid_actions(self, action):
        kwargs = dict(action_type=action, reasoning_summary="test", amount=5.0)
        if action == "cooperate":
            kwargs["target_agent_id"] = "agent_1"
        action_obj = ProposedAction(**kwargs)
        assert action_obj.action_type == action


# ── Cooperate-requires-target validation ─────────────────────────────────


class TestCooperateRequiresTarget:
    def test_cooperate_without_target_fails(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="cooperate",
                target_agent_id=None,
                amount=5.0,
                reasoning_summary="help neighbor",
            )

    def test_cooperate_with_target_succeeds(self):
        action = ProposedAction(
            action_type="cooperate",
            target_agent_id="agent_1",
            amount=5.0,
            reasoning_summary="help neighbor",
        )
        assert action.target_agent_id == "agent_1"

    def test_work_without_target_succeeds(self):
        action = ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="earn money",
        )
        assert action.target_agent_id is None


# ── Amount and confidence bounds ─────────────────────────────────────────


class TestAmountBounds:
    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="work",
                amount=-5.0,
                reasoning_summary="test",
            )

    def test_excessive_amount_rejected(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="work",
                amount=999.0,
                reasoning_summary="test",
            )

    def test_valid_amount_accepted(self):
        action = ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="test",
        )
        assert action.amount == 10.0

    def test_none_amount_accepted(self):
        """Amount can be None for backward compatibility."""
        action = ProposedAction(
            action_type="work",
            reasoning_summary="test",
        )
        assert action.amount is None


class TestConfidenceBounds:
    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="test",
                confidence=1.5,
            )

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="test",
                confidence=-0.1,
            )

    def test_valid_confidence_accepted(self):
        action = ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="test",
            confidence=0.8,
        )
        assert action.confidence == 0.8

    def test_none_confidence_accepted(self):
        action = ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="test",
        )
        assert action.confidence is None
