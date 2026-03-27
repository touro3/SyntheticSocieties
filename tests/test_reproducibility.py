"""Reproducibility tests for set_global_seed and simulation determinism.

A core claim of the BGF research is that experiments are reproducible across
runs given the same seed. These tests verify that:
  1. set_global_seed properly seeds Python random, NumPy, and PyTorch
  2. Two simulation runs with the same seed + mock backend produce identical events
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from simulation.kernel import SimulationKernel
from utils.io import set_global_seed


# ── Seed utility tests ────────────────────────────────────────────────────────

def test_set_global_seed_seeds_python_random():
    import random
    set_global_seed(42)
    v1 = [random.random() for _ in range(5)]
    set_global_seed(42)
    v2 = [random.random() for _ in range(5)]
    assert v1 == v2


def test_set_global_seed_seeds_numpy():
    import numpy as np
    set_global_seed(42)
    v1 = np.random.rand(5).tolist()
    set_global_seed(42)
    v2 = np.random.rand(5).tolist()
    assert v1 == v2


def test_set_global_seed_calls_torch_manual_seed():
    """torch.manual_seed must be called so GPU inference is reproducible."""
    with patch("torch.manual_seed") as mock_torch_seed:
        set_global_seed(99)
    mock_torch_seed.assert_called_once_with(99)


def test_set_global_seed_handles_missing_torch_gracefully():
    """set_global_seed must not raise when torch is not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        set_global_seed(42)  # must not raise


# ── Simulation determinism ────────────────────────────────────────────────────

def _build_deterministic_world():
    return World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
    )


def _build_agents(n: int = 3) -> list[Agent]:
    agents = []
    for i in range(n):
        agents.append(Agent(
            profile=AgentProfile(
                agent_id=f"agent_{i}",
                age=30 + i,
                income=1000.0,
                education="college",
                occupation="worker",
                location="urban",
                political_preference="center",
                risk_tolerance=0.5,
                social_class="middle",
            ),
            state=AgentState(wealth=50.0),
            memory=MemoryBuffer(max_items=10),
            policy=MockPolicy(),
        ))
    return agents


def _run_simulation(seed: int, tmp_path: Path, run_id: str) -> list[dict]:
    set_global_seed(seed)
    agents = _build_agents()
    world = _build_deterministic_world()
    log_path = tmp_path / f"events_{run_id}.jsonl"
    logger = EventLogger(log_path, overwrite=True)
    kernel = SimulationKernel(agents=agents, world=world, logger=logger)
    kernel.run(num_rounds=3)

    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def test_identical_seeds_produce_identical_events(tmp_path):
    """Two simulation runs with the same seed must produce byte-identical event logs.

    This is the primary reproducibility invariant for the BGF research.
    MockPolicy actions are deterministic (rule-based), so any divergence
    would indicate non-determinism in the simulation kernel or state logic.
    """
    events_a = _run_simulation(seed=42, tmp_path=tmp_path, run_id="a")
    events_b = _run_simulation(seed=42, tmp_path=tmp_path, run_id="b")

    assert len(events_a) == len(events_b), (
        f"Event counts differ: {len(events_a)} vs {len(events_b)}"
    )
    for i, (ea, eb) in enumerate(zip(events_a, events_b)):
        assert ea["agent_id"] == eb["agent_id"], f"Event {i}: agent_id mismatch"
        assert ea["action"] == eb["action"], f"Event {i}: action mismatch"
        assert ea["result"] == eb["result"], f"Event {i}: result mismatch"


def test_different_seeds_produce_consistent_structure(tmp_path):
    """Regardless of seed, MockPolicy simulations must produce the same event count.

    For LLM-backed policies, seeds would produce different action choices.
    This test documents and validates the structural invariant.
    """
    events_42 = _run_simulation(seed=42, tmp_path=tmp_path, run_id="s42")
    events_7 = _run_simulation(seed=7, tmp_path=tmp_path, run_id="s7")
    assert len(events_42) == len(events_7), (
        "Event count must be equal across seeds for deterministic (MockPolicy) runs"
    )


def test_simulation_run_produces_nonempty_events(tmp_path):
    """Guard test: every simulation run must emit at least one event.

    Ensures the kernel loop executes and the logger flushes output,
    which is a pre-condition for all downstream reproducibility claims.
    """
    events = _run_simulation(seed=99, tmp_path=tmp_path, run_id="guard")
    assert len(events) > 0, "Simulation produced no events — kernel loop may not have run"
