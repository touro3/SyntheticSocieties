"""Tests for utils.agent_factory — canonical agent/society construction."""

from __future__ import annotations

import pytest

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from utils.agent_factory import build_society


def _make_profile(agent_id: str, income: float = 1000.0) -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        age=30,
        income=income,
        education="college",
        occupation="worker",
        location="italy",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
    )


class TestBuildSociety:
    def test_returns_correct_number_of_agents(self):
        profiles = [_make_profile(f"a{i}") for i in range(5)]
        agents = build_society(profiles, MockPolicy())
        assert len(agents) == 5

    def test_all_items_are_agent_instances(self):
        profiles = [_make_profile("a0"), _make_profile("a1")]
        agents = build_society(profiles, MockPolicy())
        assert all(isinstance(a, Agent) for a in agents)

    def test_initial_wealth_defaults_to_profile_income(self):
        profiles = [_make_profile("rich", income=2000.0), _make_profile("poor", income=500.0)]
        agents = build_society(profiles, MockPolicy())
        assert agents[0].state.wealth == pytest.approx(2000.0)
        assert agents[1].state.wealth == pytest.approx(500.0)

    def test_explicit_initial_wealth_overrides_income(self):
        profiles = [_make_profile("a0", income=1000.0)]
        agents = build_society(profiles, MockPolicy(), initial_wealth=42.0)
        assert agents[0].state.wealth == pytest.approx(42.0)

    def test_profile_ids_preserved(self):
        ids = ["alice", "bob", "charlie"]
        profiles = [_make_profile(i) for i in ids]
        agents = build_society(profiles, MockPolicy())
        result_ids = [a.profile.agent_id for a in agents]
        assert result_ids == ids

    def test_memory_is_hierarchical_memory(self):
        profiles = [_make_profile("a0")]
        agents = build_society(profiles, MockPolicy())
        assert isinstance(agents[0].memory, HierarchicalMemory)

    def test_custom_memory_size(self):
        profiles = [_make_profile("a0")]
        agents = build_society(profiles, MockPolicy(), memory_size=25)
        assert agents[0].memory.max_recent == 25

    def test_policy_assigned_to_all_agents(self):
        policy = MockPolicy()
        profiles = [_make_profile(f"a{i}") for i in range(3)]
        agents = build_society(profiles, policy)
        assert all(a.policy is policy for a in agents)

    def test_empty_profiles_returns_empty_list(self):
        assert build_society([], MockPolicy()) == []
