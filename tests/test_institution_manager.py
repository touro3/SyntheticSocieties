"""Tests for InstitutionManager — no hidden side effects."""
import pytest
from conftest import make_agent
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.world_state import WorldState


@pytest.fixture
def manager():
    return InstitutionManager()


@pytest.fixture
def ws():
    return WorldState(round_id=1)


class TestExecuteNoSideEffects:
    def test_cooperate_does_not_mutate_target(self, manager, ws):
        source = make_agent("a1", wealth=100.0)
        target = make_agent("a2", wealth=50.0)
        lookup = {"a1": source, "a2": target}

        action = ProposedAction(
            action_type="cooperate",
            target_agent_id="a2",
            amount=5.0,
            reasoning_summary="help",
        )

        event = manager.execute(action, source, ws, lookup)

        # Target wealth must NOT be mutated by execute
        assert target.state.wealth == 50.0
        # But the event must carry the delta (1.5× cooperation multiplier)
        assert event["target_wealth_delta"] == 7.5  # 5.0 * 1.5
        assert event["wealth_delta"] == -5.0


class TestExecuteDeltas:
    def test_work_returns_correct_deltas(self, manager, ws):
        agent = make_agent("a1", wealth=50.0)
        action = ProposedAction(action_type="work", amount=10.0, reasoning_summary="earn")
        event = manager.execute(action, agent, ws, {"a1": agent})
        assert event["wealth_delta"] == 10.0
        assert event["stress_delta"] == 1.0

    def test_save_returns_correct_deltas(self, manager, ws):
        agent = make_agent("a1", wealth=50.0)
        action = ProposedAction(action_type="save", amount=5.0, reasoning_summary="rest")
        event = manager.execute(action, agent, ws, {"a1": agent})
        assert event["wealth_delta"] == 0.0
        assert event["stress_delta"] == -0.2

    def test_cooperate_returns_correct_deltas(self, manager, ws):
        source = make_agent("a1", wealth=100.0)
        target = make_agent("a2", wealth=50.0)
        lookup = {"a1": source, "a2": target}

        action = ProposedAction(
            action_type="cooperate", target_agent_id="a2",
            amount=7.0, reasoning_summary="help",
        )
        event = manager.execute(action, source, ws, lookup)
        assert event["wealth_delta"] == -7.0
        assert event["target_wealth_delta"] == 10.5  # 7.0 * 1.5 cooperation multiplier
        assert event["stress_delta"] == -0.1
        assert event["interaction_type"] == "cooperation"
        assert event["satisfaction_delta"] == 0.12
