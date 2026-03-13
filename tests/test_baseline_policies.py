import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from decision.random_policy import RandomPolicy
from decision.rule_based_policy import RuleBasedPolicy


def build_profile():
    return AgentProfile(
        agent_id="agent_1",
        age=30,
        income=1000,
        education="college",
        occupation="worker",
        location="italy",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
    )


def build_state(wealth: float):
    return AgentState(wealth=wealth)


def build_context(neighbors=None):
    if neighbors is None:
        neighbors = []
    return {
        "network": {
            "neighbors": neighbors
        }
    }


def test_mock_policy_returns_valid_action():
    policy = MockPolicy()
    action = policy.propose_action(
        profile=build_profile(),
        state=build_state(50.0),
        memory=MemoryBuffer(max_items=5).get_recent(),
        context=build_context(["a2"]),
        round_id=1,
    )
    assert action.action_type in {"work", "save", "cooperate"}


def test_random_policy_returns_valid_action():
    policy = RandomPolicy()
    action = policy.propose_action(
        profile=build_profile(),
        state=build_state(50.0),
        memory=[],
        context=build_context(["a2"]),
        round_id=1,
    )
    assert action.action_type in {"work", "save", "cooperate"}


def test_rule_based_policy_low_wealth_prefers_work():
    policy = RuleBasedPolicy()
    action = policy.propose_action(
        profile=build_profile(),
        state=build_state(50.0),
        memory=[],
        context=build_context(["a2"]),
        round_id=1,
    )
    assert action.action_type == "work"


def test_rule_based_policy_high_wealth_can_cooperate():
    policy = RuleBasedPolicy()
    action = policy.propose_action(
        profile=build_profile(),
        state=build_state(120.0),
        memory=[],
        context=build_context(["a2"]),
        round_id=1,
    )
    assert action.action_type == "cooperate"
    assert action.target_agent_id == "a2"
