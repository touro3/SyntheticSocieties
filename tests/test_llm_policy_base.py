"""Tests for LLMPolicyBase — shared retry, fallback, logging logic."""

from unittest.mock import MagicMock

from agents.state import AgentState
from decision.llm_policy_base import LLMPolicyBase
from decision.schemas import ProposedAction


class MockBackend:
    def __init__(self, responses):
        self._responses = iter(responses)

    def generate(self, messages, temperature=None):
        return next(self._responses)


def _make_base(responses, max_retries=2):
    backend = MockBackend(responses)
    base = LLMPolicyBase.__new__(LLMPolicyBase)
    base.backend = backend
    base.temperature = 0.7
    base.max_retries = max_retries
    base.prompt_logger = None
    return base


VALID_JSON = '{"action_type": "work", "amount": 10.0, "reasoning_summary": "earn", "confidence": 0.8}'


class TestRetryLoop:
    def test_succeeds_on_first_attempt(self):
        base = _make_base([(VALID_JSON, 0.1)])
        action, raw, latency, meta = base._generate_with_retries([{"role": "user", "content": "test"}], neighbors=[])
        assert action is not None
        assert action.action_type == "work"

    def test_retries_on_parse_failure(self):
        base = _make_base(
            [
                ("garbage output", 0.1),
                ("still garbage", 0.1),
                (VALID_JSON, 0.1),
            ]
        )
        action, raw, latency, meta = base._generate_with_retries([{"role": "user", "content": "test"}], neighbors=[])
        assert action is not None
        assert action.action_type == "work"

    def test_returns_none_after_exhausting_retries(self):
        base = _make_base(
            [
                ("garbage", 0.1),
                ("garbage", 0.1),
                ("garbage", 0.1),
            ]
        )
        action, raw, latency, meta = base._generate_with_retries([{"role": "user", "content": "test"}], neighbors=[])
        assert action is None


class TestFallback:
    def test_works_when_wealth_low(self):
        base = _make_base([])
        state = AgentState(wealth=30.0)
        action = base._fallback_action(state, neighbors=["a1"])
        assert action.action_type == "work"

    def test_cooperates_when_wealthy_with_neighbors(self):
        base = _make_base([])
        state = AgentState(wealth=150.0)
        action = base._fallback_action(state, neighbors=["a1"])
        assert action.action_type == "cooperate"
        assert action.target_agent_id == "a1"

    def test_saves_as_default(self):
        base = _make_base([])
        state = AgentState(wealth=80.0)
        action = base._fallback_action(state, neighbors=[])
        assert action.action_type == "save"


class TestPromptLogging:
    def test_log_called_when_logger_present(self):
        base = _make_base([(VALID_JSON, 0.1)])
        mock_logger = MagicMock()
        base.prompt_logger = mock_logger

        action = ProposedAction(action_type="work", amount=10.0, reasoning_summary="test")
        base._log_prompt(
            round_id=1,
            agent_id="a0",
            prompt_text="test prompt",
            raw_text=VALID_JSON,
            action=action,
            latency=0.1,
            parse_meta={},
        )
        mock_logger.log.assert_called_once()
