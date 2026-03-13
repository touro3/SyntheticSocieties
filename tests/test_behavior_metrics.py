import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from metrics.summary import cooperation_rate


def build_agent(agent_id: str, last_action: str) -> Agent:
    agent = Agent(
        profile=AgentProfile(
            agent_id=agent_id,
            age=30,
            income=1000,
            education="college",
            occupation="worker",
            location="italy",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
        ),
        state=AgentState(wealth=50.0),
        memory=MemoryBuffer(max_items=5),
        policy=MockPolicy(),
    )
    agent.state.last_action = last_action
    return agent


def test_cooperation_rate():
    agents = [
        build_agent("a1", "cooperate"),
        build_agent("a2", "work"),
        build_agent("a3", "cooperate"),
        build_agent("a4", "save"),
    ]

    rate = cooperation_rate(agents)
    assert rate == 0.5
