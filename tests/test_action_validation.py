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


def test_negative_amount_is_rejected():
    agent = Agent(
        profile=AgentProfile(
            agent_id="agent_1",
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

    manager = InstitutionManager()
    world_state = WorldState()

    action = ProposedAction(
        action_type="work",
        amount=-10.0,
        reasoning_summary="invalid negative amount",
        confidence=0.1,
    )

    result = manager.validate(action, agent, world_state)

    assert result.valid is False
    assert result.reason == "negative_amount"
