"""Integration tests: CollectiveMemory advancing through SimulationKernel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_agent

from agents.collective_memory import CollectiveMemory
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState


def _make_cooperation_kernel(event_logger):
    """Kernel where the agent always cooperates (wealth >= 100, has a neighbor).

    We set up two agents with a stub network so MockPolicy sees a neighbor
    and wealth >= 100, triggering the 'cooperate' branch.
    """
    from environment.network import NetworkManager
    from simulation.kernel import SimulationKernel

    a0 = make_agent("agent_0", wealth=150.0)
    a1 = make_agent("agent_1", wealth=150.0)
    network = NetworkManager.fully_connected(["agent_0", "agent_1"])
    world = World(
        state=WorldState(public_signal={"economy": "stable"}, prices={}, resources={}),
        institution_manager=InstitutionManager(),
        network_manager=network,
    )
    cm = CollectiveMemory()
    return SimulationKernel(agents=[a0, a1], world=world, logger=event_logger, collective_memory=cm), cm


def test_kernel_records_cooperation_fact_in_collective_memory(tmp_path):
    """When ≥60% of agents cooperate, collective memory should gain a cooperation fact."""
    from bgf_logging.event_logger import EventLogger

    log = EventLogger(tmp_path / "ev.jsonl", overwrite=True)
    kernel, cm = _make_cooperation_kernel(log)
    kernel.run_round()
    kernel._log_round_metrics()

    facts = cm.snapshot()
    fact_types = {f.fact_type for f in facts}
    assert "norm_shift" in fact_types


def test_collective_memory_decays_across_rounds(minimal_kernel, collective_memory):
    """A fact recorded at importance=1.0 should decay below 0.5 after enough rounds."""
    collective_memory.record(0, "test", "Initial fact.", importance=1.0)

    # Advance 10 rounds (half_life_rounds default = 10 → importance halves once)
    for _ in range(10):
        minimal_kernel.run_round()

    facts = collective_memory.snapshot()
    assert facts, "Fact should still exist (not yet pruned)"
    assert facts[0].importance < 0.55


def test_agent_contributes_fact_to_collective_memory(collective_memory):
    """CollectiveMemory.contribute() should add an agent_belief fact."""
    collective_memory.contribute("agent_42", round_id=3, fact_text="Trust is declining in my neighborhood.")

    facts = collective_memory.snapshot()
    assert len(facts) == 1
    assert facts[0].fact_type == "agent_belief"
    assert "agent_42" in facts[0].content
