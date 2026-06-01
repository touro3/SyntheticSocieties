"""Tests for bias & failure diagnostics — no GPU required."""

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from metrics.diagnostics import (
    alignment_bias_detection,
    persona_drift_detection,
    response_diversity,
    subgroup_analysis,
)


def _make_agents():
    policy = MockPolicy()
    agents = []
    for i, (trust, gender) in enumerate([(0.8, 1), (0.2, 2), (0.5, 1), (0.9, 2)]):
        profile = AgentProfile(
            agent_id=f"agent_{i}",
            age=30 + i * 5,
            income=1000.0,
            education="college",
            occupation="worker",
            location="urban",
            political_preference="center",
            social_class="middle",
            trust_people=trust,
            risk_tolerance=0.5,
            gender=gender,
        )
        state = AgentState(wealth=50.0 + i * 20)
        agents.append(Agent(profile=profile, state=state, memory=MemoryBuffer(), policy=policy))
    return agents


def _make_events():
    events = []
    for r in range(5):
        for i in range(4):
            action = ["work", "cooperate", "save", "work"][i]
            events.append(
                {
                    "round_id": r,
                    "agent_id": f"agent_{i}",
                    "action": {"action_type": action},
                }
            )
    return events


def test_subgroup_analysis_by_gender():
    agents = _make_agents()
    events = _make_events()
    result = subgroup_analysis(events, agents, group_by="gender")
    assert "1" in result  # male
    assert "2" in result  # female
    assert result["1"]["n_agents"] == 2
    assert result["2"]["n_agents"] == 2


def test_persona_drift_detection():
    events = _make_events()
    agents = _make_agents()
    result = persona_drift_detection(events, agents, window_size=2)
    assert "agent_0" in result
    assert "drift_jsd" in result["agent_0"]


def test_response_diversity():
    events = _make_events()
    result = response_diversity(events)
    assert "agent_0" in result
    assert "entropy" in result["agent_0"]
    assert "_aggregate" in result
    # Agent 0 only does "work" → entropy = 0
    assert result["agent_0"]["entropy"] == 0.0


def test_alignment_bias_detection():
    agents = _make_agents()
    events = _make_events()
    result = alignment_bias_detection(events, agents)
    assert "trust_cooperation_correlation" in result
    assert "persona_responsive" in result
    assert "potential_bias" in result
