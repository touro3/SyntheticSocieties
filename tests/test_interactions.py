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

    # But deltas must be correct in the event dict (1.5× cooperation multiplier)
    assert event["wealth_delta"] == -5.0
    assert event["target_wealth_delta"] == 7.5  # 5.0 * 1.5

    # The kernel is responsible for applying target_wealth_delta
    source.apply_local_update(event)
    assert source.state.wealth == 95.0

    # Simulate kernel applying target delta
    target.state.wealth += event["target_wealth_delta"]
    assert target.state.wealth == 57.5  # 50.0 + 7.5


# ── Peer-to-Peer Communication (Information Contagion) ────────────────────────

from unittest.mock import MagicMock

from agents.memory import HierarchicalMemory
from simulation.round_processor import RoundProcessor


class TestCommunicateAction:
    """Communicate action injects gossip into target agent's memory."""

    def _make_agents(self):
        a = make_agent("sender", wealth=100.0)
        a.memory = HierarchicalMemory()
        b = make_agent("receiver", wealth=100.0)
        b.memory = HierarchicalMemory()
        return a, b

    def test_communicate_creates_memory_in_target(self):
        """Agent A communicate(target=B, msg) → B receives MemoryItem."""
        sender, receiver = self._make_agents()
        agent_lookup = {"sender": sender, "receiver": receiver}

        world = MagicMock()
        world.validate_action.return_value = MagicMock(valid=True)
        world.execute_action.return_value = {
            "agent_id": "sender",
            "action_type": "communicate",
            "target_agent_id": "receiver",
            "message_summary": "I plan to cooperate next round",
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "satisfaction_delta": 0.0,
            "target_wealth_delta": 0.0,
            "interaction_type": "communication",
            "round_id": 5,
        }
        logger = MagicMock()
        processor = RoundProcessor(world=world, agent_lookup=agent_lookup, logger=logger)

        action = ProposedAction(
            action_type="communicate",
            target_agent_id="receiver",
            reasoning_summary="I plan to cooperate next round",
        )

        processor.process_agent_action(sender, action, round_id=5)

        # Receiver should have a memory item from the communication
        comm_items = [m for m in receiver.memory.recent if m.event_type == "received_message"]
        assert len(comm_items) == 1
        item = comm_items[0]
        assert "sender" in item.content or item.partner_id == "sender"
        assert item.round_id == 5

    def test_communicate_tags_source_in_memory(self):
        """The memory item must record the source agent ID."""
        sender, receiver = self._make_agents()
        agent_lookup = {"sender": sender, "receiver": receiver}

        world = MagicMock()
        world.validate_action.return_value = MagicMock(valid=True)
        world.execute_action.return_value = {
            "agent_id": "sender",
            "action_type": "communicate",
            "target_agent_id": "receiver",
            "message_summary": "Cooperation is beneficial",
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "satisfaction_delta": 0.0,
            "target_wealth_delta": 0.0,
            "interaction_type": "communication",
            "round_id": 3,
        }
        logger = MagicMock()
        processor = RoundProcessor(world=world, agent_lookup=agent_lookup, logger=logger)

        action = ProposedAction(
            action_type="communicate",
            target_agent_id="receiver",
            reasoning_summary="Cooperation is beneficial",
        )

        processor.process_agent_action(sender, action, round_id=3)

        comm_items = [m for m in receiver.memory.recent if m.event_type == "received_message"]
        assert len(comm_items) == 1
        assert comm_items[0].partner_id == "sender"

    def test_communicate_does_not_change_wealth(self):
        """Communication has no economic effect."""
        sender, receiver = self._make_agents()
        agent_lookup = {"sender": sender, "receiver": receiver}

        world = MagicMock()
        world.validate_action.return_value = MagicMock(valid=True)
        world.execute_action.return_value = {
            "agent_id": "sender",
            "action_type": "communicate",
            "target_agent_id": "receiver",
            "message_summary": "gossip",
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "satisfaction_delta": 0.0,
            "target_wealth_delta": 0.0,
            "interaction_type": "communication",
            "round_id": 1,
        }
        logger = MagicMock()
        processor = RoundProcessor(world=world, agent_lookup=agent_lookup, logger=logger)

        action = ProposedAction(
            action_type="communicate",
            target_agent_id="receiver",
            reasoning_summary="gossip",
        )

        processor.process_agent_action(sender, action, round_id=1)

        assert sender.state.wealth == 100.0
        assert receiver.state.wealth == 100.0

    def test_communicate_schema_accepted(self):
        """ProposedAction accepts 'communicate' as a valid action_type."""
        action = ProposedAction(
            action_type="communicate",
            target_agent_id="agent_1",
            reasoning_summary="sharing information",
        )
        assert action.action_type == "communicate"

    def test_parser_handles_communicate(self):
        """output_parser.parse_llm_output handles communicate action."""
        from decision.output_parser import parse_llm_output

        raw = '{"action_type": "communicate", "target_agent_id": "agent_1", "reasoning_summary": "gossip"}'
        result, meta = parse_llm_output(raw, neighbors=["agent_1"])
        assert result is not None
        assert result.action_type == "communicate"
        assert result.target_agent_id == "agent_1"
