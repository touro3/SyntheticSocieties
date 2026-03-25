"""Tests for cooperation wealth transfer — via kernel, not side-effects."""
from conftest import make_agent
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.world_state import WorldState


def test_cooperate_transfer_via_event_dict():
    """InstitutionManager.execute returns correct deltas without mutating target."""
    source = make_agent("a1", wealth=100.0)
    target = make_agent("a2", wealth=50.0)
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

    # execute() must NOT mutate target directly
    assert target.state.wealth == 50.0

    # But deltas must be correct in the event dict
    assert event["wealth_delta"] == -5.0
    assert event["target_wealth_delta"] == 5.0

    # The kernel is responsible for applying target_wealth_delta
    source.apply_local_update(event)
    assert source.state.wealth == 95.0

    # Simulate kernel applying target delta
    target.state.wealth += event["target_wealth_delta"]
    assert target.state.wealth == 55.0
