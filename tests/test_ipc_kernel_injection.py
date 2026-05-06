"""Integration tests: IPC injections consumed by SimulationKernel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from conftest import make_agent


def test_kernel_consumes_wealth_shock_injection_in_round(minimal_kernel):
    """A wealth_shock injection queued on world_state should be drained in one round."""
    world = minimal_kernel.world
    world.state.pending_injections.append(
        {"event_type": "wealth_shock", "payload": {"magnitude": 20.0}}
    )
    minimal_kernel.run_round()

    assert world.state.shock_active is True
    assert world.state.shock_magnitude == pytest.approx(20.0)
    assert world.state.pending_injections == []


def test_kernel_consumes_narrative_injection_in_round(minimal_kernel, collective_memory):
    """A narrative injection should be recorded in collective memory after one round."""
    world = minimal_kernel.world
    world.state.pending_injections.append(
        {"event_type": "narrative", "payload": {"content": "A major policy shift occurred."}}
    )
    minimal_kernel.run_round()

    facts = collective_memory.snapshot()
    contents = [f.content for f in facts]
    assert any("policy shift" in c for c in contents)


def test_multiple_injections_drain_after_single_round(minimal_kernel):
    """All queued injections should be drained (pending_injections empty) after one round."""
    world = minimal_kernel.world
    world.state.pending_injections.extend(
        [
            {"event_type": "signal_update", "payload": {"signal": {"inflation": "high"}}},
            {"event_type": "narrative", "payload": {"content": "Market tension rising."}},
        ]
    )
    assert len(world.state.pending_injections) == 2
    minimal_kernel.run_round()
    assert world.state.pending_injections == []
