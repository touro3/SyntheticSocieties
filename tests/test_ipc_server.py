"""Tests for SimulationIPCServer and SimulationIPCClient (interview side)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from conftest import make_agent

from simulation.ipc import SimulationIPCClient, SimulationIPCServer


@pytest.fixture()
def ipc_pair(tmp_path):
    """Start a server+client pair sharing tmp_path, yield (server, client)."""
    agent_a = make_agent("agent_a", wealth=42.0)
    agent_b = make_agent("agent_b", wealth=10.0)
    registry = {a.profile.agent_id: a for a in (agent_a, agent_b)}

    from environment.world_state import WorldState

    world_state = WorldState()
    world_state.pending_injections = []

    server = SimulationIPCServer(
        agents=registry,
        base_dir=tmp_path,
        current_round_fn=lambda: 7,
        world_state=world_state,
    )
    server.start()
    client = SimulationIPCClient(base_dir=tmp_path, timeout=5.0)
    yield server, client, registry, world_state
    server.stop()


def test_interview_agent_returns_state(ipc_pair):
    """interview_agent command returns a string describing the agent's state."""
    _, client, _, _ = ipc_pair
    result = client.interview_agent("agent_a", "What is your current wealth?")
    answer = result.get("answer", "")
    assert "42" in answer or "wealth" in answer.lower()
    assert result.get("agent_id") == "agent_a"


def test_interview_agent_unknown_id_returns_error(ipc_pair):
    """Querying a non-existent agent returns an error dict."""
    _, client, _, _ = ipc_pair
    result = client.interview_agent("ghost_agent", "What are you?")
    assert "error" in result


def test_interview_batch_returns_multiple(ipc_pair):
    """interview_batch returns answers for all requested agent IDs."""
    _, client, _, _ = ipc_pair
    result = client.interview_batch(["agent_a", "agent_b"], "Summarise your decisions.")
    responses = result.get("responses", {})
    assert "agent_a" in responses
    assert "agent_b" in responses


def test_list_agents_returns_all(ipc_pair):
    """list_agents returns info for every registered agent."""
    _, client, registry, _ = ipc_pair
    agents_info = client.list_agents()
    returned_ids = {a["agent_id"] for a in agents_info}
    assert returned_ids == set(registry.keys())
    # Each entry should include wealth
    for entry in agents_info:
        assert "wealth" in entry


def test_get_status_returns_round(ipc_pair):
    """get_status returns the current round from the provided current_round_fn."""
    _, client, _, _ = ipc_pair
    status = client.get_status()
    assert status["current_round"] == 7
    assert status["n_agents"] == 2


def test_unknown_command_returns_error(ipc_pair):
    """Sending an unrecognised command returns an error result."""
    _, client, _, _ = ipc_pair
    result = client.send("not_a_real_command", {})
    assert "error" in result


def test_inject_event_queues_on_world_state(ipc_pair):
    """inject_event command appends to world_state.pending_injections."""
    _, client, _, world_state = ipc_pair
    result = client.inject_event("narrative", {"content": "Market crash imminent."})
    assert result.get("status") == "ok"
    assert len(world_state.pending_injections) == 1
    assert world_state.pending_injections[0]["event_type"] == "narrative"
