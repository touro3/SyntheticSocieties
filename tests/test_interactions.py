import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.world_state import WorldState


def build_agent(agent_id: str, wealth: float) -> Agent:
    return Agent(
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
        state=AgentState(wealth=wealth),
        memory=MemoryBuffer(max_items=5),
        policy=MockPolicy(),
    )


def test_cooperate_transfer_execution():
    source = build_agent("a1", 100.0)
    target = build_agent("a2", 50.0)
    manager = InstitutionManager()
    world_state = WorldState()

    action = ProposedAction(
        action_type="cooperate",
        target_agent_id="a2",
        amount=5.0,
        reasoning_summary="help neighbor",
        confidence=0.8,
    )

    result = manager.validate(action, source, world_state, {"a1": source, "a2": target})
    assert result.valid is True

    event = manager.execute(action, source, world_state, {"a1": source, "a2": target})

    assert event["wealth_delta"] == -5.0
    assert event["target_wealth_delta"] == 5.0
    assert target.state.wealth == 55.0
