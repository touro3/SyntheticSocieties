"""Tests for decision/rule_based_ess_policy.py.2."""

from __future__ import annotations

from decision.constants import (
    STRESS_CRITICAL,
    WORK_WEALTH_THRESHOLD,
)
from decision.rule_based_ess_policy import (
    RuleBasedESSPolicy,
    _cooperation_probability,
    _deterministic_uniform,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeProfile:
    def __init__(
        self,
        agent_id="agent_test",
        trust_people=0.6,
        trust_institutions=0.5,
        risk_tolerance=0.3,
        social_activity=0.5,
        **kwargs,
    ):
        self.agent_id = agent_id
        self.trust_people = trust_people
        self.trust_institutions = trust_institutions
        self.risk_tolerance = risk_tolerance
        self.social_activity = social_activity
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeState:
    def __init__(self, wealth=100.0, stress=0.2):
        self.wealth = wealth
        self.stress = stress


def _ctx_with_neighbors(neighbor_ids: list[str]) -> dict:
    return {"network": {"neighbors": neighbor_ids}}


# ── _deterministic_uniform ────────────────────────────────────────────────────


class TestDeterministicUniform:
    def test_range(self):
        for agent_id in ["a0", "a1", "a99"]:
            for r in range(10):
                val = _deterministic_uniform(agent_id, r)
                assert 0.0 <= val < 1.0

    def test_reproducible(self):
        v1 = _deterministic_uniform("agent_42", 7)
        v2 = _deterministic_uniform("agent_42", 7)
        assert v1 == v2

    def test_different_inputs_different_values(self):
        v1 = _deterministic_uniform("agent_1", 1)
        v2 = _deterministic_uniform("agent_2", 1)
        assert v1 != v2


# ── _cooperation_probability ──────────────────────────────────────────────────


class TestCooperationProbability:
    def test_high_trust_low_risk_high_prob(self):
        p = _FakeProfile(trust_people=0.9, risk_tolerance=0.1, social_activity=0.8)
        prob = _cooperation_probability(p)
        assert prob > 0.5

    def test_low_trust_high_risk_low_prob(self):
        p = _FakeProfile(trust_people=0.1, risk_tolerance=0.9, social_activity=0.1)
        prob = _cooperation_probability(p)
        assert prob < 0.4

    def test_clipped_to_floor(self):
        p = _FakeProfile(trust_people=0.0, risk_tolerance=1.0, social_activity=0.0)
        prob = _cooperation_probability(p)
        assert prob >= 0.05

    def test_clipped_to_ceiling(self):
        p = _FakeProfile(trust_people=1.0, risk_tolerance=0.0, social_activity=1.0)
        prob = _cooperation_probability(p)
        assert prob <= 0.90

    def test_none_attributes_default_to_midpoint(self):
        p = _FakeProfile(trust_people=None, risk_tolerance=None, social_activity=None)
        prob = _cooperation_probability(p)
        # 0.2 + 0.5*0.5*0.5 + 0.15*0.5 = 0.2 + 0.125 + 0.075 = 0.4
        assert abs(prob - 0.4) < 0.01


# ── RuleBasedESSPolicy ────────────────────────────────────────────────────────


class TestRuleBasedESSPolicy:
    def _policy(self):
        return RuleBasedESSPolicy()

    def test_returns_proposed_action(self):
        from decision.schemas import ProposedAction

        policy = self._policy()
        profile = _FakeProfile()
        state = _FakeState(wealth=120.0)
        action = policy.propose_action(profile, state, None, _ctx_with_neighbors(["b1"]), 0)
        assert isinstance(action, ProposedAction)

    def test_poverty_escape_returns_work(self):
        policy = self._policy()
        profile = _FakeProfile(trust_people=0.9, risk_tolerance=0.1)  # high coop prob
        state = _FakeState(wealth=WORK_WEALTH_THRESHOLD - 1.0)
        action = policy.propose_action(profile, state, None, _ctx_with_neighbors(["b1"]), 0)
        assert action.action_type == "work"

    def test_high_stress_returns_work(self):
        policy = self._policy()
        profile = _FakeProfile(trust_people=0.9)
        state = _FakeState(wealth=200.0, stress=STRESS_CRITICAL + 0.1)
        action = policy.propose_action(profile, state, None, _ctx_with_neighbors(["b1"]), 0)
        assert action.action_type == "work"

    def test_action_type_valid(self):
        valid = {"work", "save", "cooperate"}
        policy = self._policy()
        for i in range(20):
            profile = _FakeProfile(agent_id=f"agent_{i}")
            state = _FakeState(wealth=150.0, stress=0.2)
            action = policy.propose_action(profile, state, None, _ctx_with_neighbors(["neighbor_0"]), i)
            assert action.action_type in valid

    def test_no_neighbors_no_cooperate(self):
        """Without neighbors there is no valid cooperation target."""
        policy = self._policy()
        # Use an agent_id / round that would hash to cooperate range
        profile = _FakeProfile(trust_people=0.95, risk_tolerance=0.05, social_activity=0.9)
        state = _FakeState(wealth=200.0)
        # Over many rounds without neighbors, never cooperate
        actions = set()
        for r in range(50):
            a = policy.propose_action(profile, state, None, _ctx_with_neighbors([]), r)
            actions.add(a.action_type)
        assert "cooperate" not in actions

    def test_reproducible_same_inputs(self):
        policy = self._policy()
        profile = _FakeProfile(agent_id="fixed_agent")
        state = _FakeState()
        ctx = _ctx_with_neighbors(["n1"])
        a1 = policy.propose_action(profile, state, None, ctx, 5)
        a2 = policy.propose_action(profile, state, None, ctx, 5)
        assert a1.action_type == a2.action_type

    def test_different_rounds_can_differ(self):
        """Different rounds may produce different actions for the same agent."""
        policy = self._policy()
        profile = _FakeProfile(agent_id="agent_sweep", trust_people=0.5, risk_tolerance=0.5)
        state = _FakeState(wealth=150.0)
        ctx = _ctx_with_neighbors(["n1"])
        actions = {policy.propose_action(profile, state, None, ctx, r).action_type for r in range(30)}
        # With a mid-range profile, expect to see at least 2 different actions
        assert len(actions) >= 2

    def test_cooperate_sets_target(self):
        """Cooperate action must have a target_agent_id."""
        policy = self._policy()
        profile = _FakeProfile(agent_id="agent_0", trust_people=0.9)
        state = _FakeState(wealth=200.0)
        ctx = _ctx_with_neighbors(["n1", "n2"])
        found_cooperate = False
        for r in range(50):
            a = policy.propose_action(profile, state, None, ctx, r)
            if a.action_type == "cooperate":
                assert a.target_agent_id is not None
                found_cooperate = True
                break
        # High-trust agent should cooperate at least once in 50 rounds
        assert found_cooperate, "Expected at least one cooperate in 50 rounds for high-trust agent"

    def test_confidence_is_high(self):
        policy = self._policy()
        profile = _FakeProfile()
        state = _FakeState()
        action = policy.propose_action(profile, state, None, _ctx_with_neighbors(["n1"]), 0)
        assert action.confidence >= 0.8

    def test_reasoning_summary_non_empty(self):
        policy = self._policy()
        profile = _FakeProfile()
        state = _FakeState()
        action = policy.propose_action(profile, state, None, _ctx_with_neighbors(["n1"]), 0)
        assert action.reasoning_summary and len(action.reasoning_summary) > 0
