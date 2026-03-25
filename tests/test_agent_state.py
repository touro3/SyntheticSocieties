"""Tests for AgentState validation and clamping."""
from agents.state import AgentState


class TestAgentStateClamping:
    def test_wealth_clamps_at_zero(self):
        state = AgentState(wealth=-5.0)
        assert state.wealth == 0.0

    def test_positive_wealth_unchanged(self):
        state = AgentState(wealth=100.0)
        assert state.wealth == 100.0

    def test_zero_wealth_accepted(self):
        state = AgentState(wealth=0.0)
        assert state.wealth == 0.0

    def test_clamp_method_normalizes_state(self):
        state = AgentState(wealth=50.0, stress=2.0, satisfaction=-0.5)
        state.clamp()
        assert state.stress <= 1.0
        assert state.satisfaction >= 0.0

    def test_clamp_leaves_valid_state_unchanged(self):
        state = AgentState(wealth=50.0, stress=0.3, satisfaction=0.7)
        state.clamp()
        assert state.wealth == 50.0
        assert state.stress == 0.3
        assert state.satisfaction == 0.7


class TestAgentStateSnapshot:
    def test_snapshot_returns_dict(self):
        state = AgentState(wealth=50.0, stress=0.3, satisfaction=0.7, last_action="work")
        snap = state.snapshot()
        assert snap == {
            "wealth": 50.0,
            "stress": 0.3,
            "satisfaction": 0.7,
            "last_action": "work",
        }

    def test_snapshot_is_independent_copy(self):
        state = AgentState(wealth=50.0)
        snap = state.snapshot()
        state.wealth = 100.0
        assert snap["wealth"] == 50.0
