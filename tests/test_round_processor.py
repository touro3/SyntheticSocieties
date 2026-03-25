"""Tests for RoundProcessor — extracted from SimulationKernel."""
from conftest import make_agent
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from bgf_logging.event_logger import EventLogger
from simulation.round_processor import RoundProcessor


def _setup(tmp_path):
    a1 = make_agent("a1", wealth=100.0)
    a2 = make_agent("a2", wealth=50.0)
    world = World(
        state=WorldState(round_id=1),
        institution_manager=InstitutionManager(),
    )
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)
    agent_lookup = {"a1": a1, "a2": a2}
    processor = RoundProcessor(world=world, agent_lookup=agent_lookup, logger=logger)
    return processor, a1, a2


class TestProcessValidAction:
    def test_work_action_returns_event(self, tmp_path):
        proc, a1, _ = _setup(tmp_path)
        action = ProposedAction(action_type="work", amount=10.0, reasoning_summary="earn")
        event = proc.process_agent_action(a1, action, round_id=1)
        assert event["action_type"] == "work"
        assert event["wealth_delta"] == 10.0

    def test_work_updates_agent_state(self, tmp_path):
        proc, a1, _ = _setup(tmp_path)
        action = ProposedAction(action_type="work", amount=10.0, reasoning_summary="earn")
        proc.process_agent_action(a1, action, round_id=1)
        assert a1.state.wealth == 110.0


class TestProcessCooperation:
    def test_cooperate_applies_target_delta(self, tmp_path):
        proc, a1, a2 = _setup(tmp_path)
        action = ProposedAction(
            action_type="cooperate", target_agent_id="a2",
            amount=5.0, reasoning_summary="help",
        )
        proc.process_agent_action(a1, action, round_id=1)
        assert a1.state.wealth == 95.0
        assert a2.state.wealth == 55.0


class TestProcessRejection:
    def test_rejected_action_returns_rejection_event(self, tmp_path):
        proc, a1, _ = _setup(tmp_path)
        # Cooperate with self → rejected
        action = ProposedAction(
            action_type="cooperate", target_agent_id="a1",
            amount=5.0, reasoning_summary="self help",
        )
        event = proc.process_agent_action(a1, action, round_id=1)
        assert event["action_type"] == "rejected"
        assert event["reason"] == "self_target_not_allowed"


class TestProcessMemory:
    def test_records_memory_for_valid_action(self, tmp_path):
        proc, a1, _ = _setup(tmp_path)
        action = ProposedAction(action_type="work", amount=10.0, reasoning_summary="earn")
        proc.process_agent_action(a1, action, round_id=1)
        assert len(a1.memory.recent) == 1
        assert a1.memory.recent[0].event_type == "work"
