import pytest
from pydantic import ValidationError

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.world_state import WorldState


def test_negative_amount_is_rejected_at_schema_level():
    """Negative amounts are now caught by Pydantic validation at construction time."""
    with pytest.raises(ValidationError):
        ProposedAction(
            action_type="work",
            amount=-10.0,
            reasoning_summary="invalid negative amount",
            confidence=0.1,
        )
