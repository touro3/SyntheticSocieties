"""Agent construction utilities — single source of truth for society building.

Consolidates the `build_society()` pattern that was duplicated across
run_bad_apple.py, run_topology.py, run_macro_shock.py, and run_phase_d_scaling.py.
"""

from __future__ import annotations

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState


def build_society(
    profiles: list[AgentProfile],
    policy,
    *,
    memory_size: int = 10,
    initial_wealth: float | None = None,
) -> list[Agent]:
    """Create a list of agents from empirical profiles.

    Args:
        profiles: Agent profiles (e.g. from EmpiricalProfileLoader).
        policy: Decision policy applied to every agent.
        memory_size: HierarchicalMemory capacity (max recent items).
        initial_wealth: Starting wealth; defaults to profile.income if None.

    Returns:
        List of fully initialised Agent instances.
    """
    return [
        Agent(
            profile=p,
            state=AgentState(wealth=initial_wealth if initial_wealth is not None else p.income),
            memory=HierarchicalMemory(max_recent=memory_size),
            policy=policy,
        )
        for p in profiles
    ]
