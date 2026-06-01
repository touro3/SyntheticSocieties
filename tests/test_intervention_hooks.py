"""Tests for causal-diagnostic intervention hooks.

Guarantees:
  - trust_shock lowers per-partner trust on agents;
  - scarcity / wealth_shock are queued onto the existing world injection
    pathway and the unmodified World applies them;
  - hooks fire only at the trigger round (unless repeat=True);
  - the betrayal-memory counterfactual removes exactly the targeted items.
"""

from __future__ import annotations

import pytest

from agents.memory import HierarchicalMemory, MemoryItem
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from simulation.intervention_hooks import (
    InterventionHook,
    InterventionHookRegistry,
    delete_betrayal_memories,
)
from tests.conftest import make_agent


def _world() -> World:
    return World(
        state=WorldState(public_signal={"economy": "stable"}, prices={"food": 1.0}, resources={"jobs": 100.0}),
        institution_manager=InstitutionManager(),
    )


def test_invalid_kind_rejected():
    with pytest.raises(ValueError, match="Unknown intervention kind"):
        InterventionHook(trigger_round=1, kind="teleport")


def test_trust_shock_lowers_trust():
    a = make_agent("agent_0")
    a.state.trust = {"agent_1": 0.8, "agent_2": 0.6}
    reg = InterventionHookRegistry([InterventionHook(2, "trust_shock", {"magnitude": 0.5})])

    assert reg.apply(1, [a]) == []  # not due yet
    assert a.state.trust["agent_1"] == 0.8

    events = reg.apply(2, [a])
    assert len(events) == 1
    assert a.state.trust["agent_1"] == pytest.approx(0.4)
    assert a.state.trust["agent_2"] == pytest.approx(0.3)
    assert events[0]["n_trust_edges_affected"] == 2


def test_trust_shock_partner_scoped():
    a = make_agent("agent_0")
    a.state.trust = {"agent_1": 0.8, "agent_2": 0.6}
    reg = InterventionHookRegistry([InterventionHook(0, "trust_shock", {"magnitude": 1.0, "partner": "agent_1"})])
    reg.apply(0, [a])
    assert a.state.trust["agent_1"] == 0.0
    assert a.state.trust["agent_2"] == 0.6  # untouched


def test_scarcity_queued_and_applied_by_world():
    w = _world()
    reg = InterventionHookRegistry([InterventionHook(0, "scarcity", {"severity": 0.5})])
    reg.apply(0, [], world=w)
    assert len(w.state.pending_injections) == 1
    w.apply_exogenous_updates()  # unmodified world consumes the queue
    assert w.state.public_signal.get("scarcity") == "true"
    assert w.state.resources["jobs"] == pytest.approx(50.0)


def test_wealth_shock_queued():
    w = _world()
    reg = InterventionHookRegistry([InterventionHook(0, "wealth_shock", {"magnitude": 0.5})])
    reg.apply(0, [], world=w)
    w.apply_exogenous_updates()
    assert w.state.shock_active is True


def test_scarcity_requires_world():
    reg = InterventionHookRegistry([InterventionHook(0, "scarcity", {})])
    with pytest.raises(ValueError, match="requires a world"):
        reg.apply(0, [])


def test_repeat_fires_every_round_after_trigger():
    a = make_agent("agent_0")
    a.state.trust = {"agent_1": 1.0}
    reg = InterventionHookRegistry([InterventionHook(1, "trust_shock", {"magnitude": 0.1}, repeat=True)])
    assert reg.apply(0, [a]) == []
    assert reg.apply(1, [a])
    assert reg.apply(2, [a])
    assert a.state.trust["agent_1"] < 0.9


def test_delete_betrayal_memories_targeted():
    mem = HierarchicalMemory(max_recent=20)
    mem.add(MemoryItem(0, "agent_0", "cooperate", "betrayed", {"reciprocated": False}, importance=0.9))
    mem.add(MemoryItem(1, "agent_0", "cooperate", "ok", {"reciprocated": True}, importance=0.5))
    mem.add(MemoryItem(2, "agent_9", "cooperate", "other betrayal", {"reciprocated": False}, importance=0.9))
    mem.add(MemoryItem(3, None, "work", "worked", {}, importance=0.1))

    class _A:
        pass

    a = _A()
    a.memory = mem

    removed = delete_betrayal_memories(a, partner_id="agent_0")
    assert removed == 1
    remaining = [(it.partner_id, it.event_type, it.outcome.get("reciprocated")) for it in mem.recent]
    assert ("agent_0", "cooperate", False) not in remaining
    assert ("agent_9", "cooperate", False) in remaining  # other partner kept
    assert len(mem.recent) == 3
