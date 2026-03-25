"""Formal interface contract for all agent decision policies.

Using Protocol (PEP 544) for structural subtyping — existing policy classes
automatically conform without changing their class declarations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agents.profile import AgentProfile
from agents.state import AgentState
from agents.memory import HierarchicalMemory
from decision.schemas import ProposedAction


@runtime_checkable
class PolicyProtocol(Protocol):
    """All agent policies must implement this interface."""

    def propose_action(
        self,
        profile: AgentProfile,
        state: AgentState,
        memory: HierarchicalMemory,
        context: dict,
        round_id: int,
    ) -> ProposedAction: ...
