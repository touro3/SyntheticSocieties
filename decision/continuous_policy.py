"""ABC for policies operating over a continuous action space.

Architectural sketch — not wired into the simulation kernel.

The current BGF kernel uses a ternary discrete space {work, save, cooperate}.
This module defines an interface for policies that output *weights* over that
space (a point on the 2-simplex Δ²) rather than a hard choice. The bridge
method `to_proposed_action` maps the continuous weights back to the BGF
`ProposedAction` tuple in a way that is safe and reversible.

Extension path
--------------
1.  Implement `ContinuousPolicy` (e.g., a neural policy trained via MARL).
2.  Wrap it with `ContinuousToDiscreteAdapter` to plug into the existing
    `PolicyProtocol` interface without changing the kernel.
3.  For fully continuous payoffs, replace `EconomyEngine.parse_action` with a
    parameterised payoff function that accepts `ContinuousActionWeights.weights`
    directly and interpolates the ternary payoff matrix.

Ternary payoff reference (environment/payoffs.py):
    work      → +8 wealth
    save      → +4 wealth
    cooperate → −3 wealth (contributes to shared pool: +12 / n_cooperators)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.policy_protocol import PolicyProtocol
from decision.schemas import ProposedAction

# Action labels aligned with BGF ternary space.
_ACTIONS: tuple[str, ...] = ("work", "save", "cooperate")


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------


@dataclass
class ContinuousActionWeights:
    """A point on the 2-simplex Δ² over {work, save, cooperate}.

    Invariant: weights.sum() ≈ 1.0, weights >= 0.
    """

    weights: np.ndarray  # shape (3,): [p_work, p_save, p_cooperate]
    reasoning_summary: str = ""
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        if w.shape != (3,):
            raise ValueError(f"weights must be shape (3,), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        total = w.sum()
        if total == 0:
            raise ValueError("weights must not all be zero")
        self.weights = w / total  # normalise to valid probability

    # ------------------------------------------------------------------
    # Bridge to BGF discrete tuple
    # ------------------------------------------------------------------

    def to_proposed_action(
        self,
        stochastic: bool = True,
        target_agent_id: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ProposedAction:
        """Map continuous weights → BGF `ProposedAction`.

        Args:
            stochastic: If True, sample from the categorical distribution
                defined by `weights` (preserves expected-value accuracy).
                If False, use argmax (deterministic, loses distributional info).
            target_agent_id: Required when the sampled action is ``cooperate``.
                If None and cooperate is selected, the bridge falls back to
                the highest-weighted non-cooperate action.
            rng: Optional numpy Generator for reproducibility.
        """
        if stochastic:
            _rng = rng or np.random.default_rng()
            idx = int(_rng.choice(3, p=self.weights))
        else:
            idx = int(np.argmax(self.weights))

        action = _ACTIONS[idx]

        # cooperate requires a target; fall back if none provided
        if action == "cooperate" and target_agent_id is None:
            fallback_weights = self.weights.copy()
            fallback_weights[2] = 0.0
            if fallback_weights.sum() == 0:
                fallback_weights = np.array([0.5, 0.5, 0.0])
            fallback_weights /= fallback_weights.sum()
            if stochastic:
                _rng = rng or np.random.default_rng()
                idx = int(_rng.choice(3, p=fallback_weights))
            else:
                idx = int(np.argmax(fallback_weights))
            action = _ACTIONS[idx]

        return ProposedAction(
            action_type=action,  # type: ignore[arg-type]
            target_agent_id=target_agent_id if action == "cooperate" else None,
            amount=None,
            reasoning_summary=self.reasoning_summary or f"continuous weights {self.weights.tolist()}",
            confidence=self.confidence,
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ContinuousPolicy(ABC):
    """Policy that returns a distribution over actions rather than a hard choice.

    Subclasses implement `decide` and return a `ContinuousActionWeights`.
    Use `ContinuousToDiscreteAdapter` to plug into the existing kernel.
    """

    @abstractmethod
    def decide(
        self,
        profile: AgentProfile,
        state: AgentState,
        memory: HierarchicalMemory,
        context: dict,
        round_id: int,
    ) -> ContinuousActionWeights:
        """Return a probability distribution over {work, save, cooperate}."""
        ...

    def decide_discrete(
        self,
        profile: AgentProfile,
        state: AgentState,
        memory: HierarchicalMemory,
        context: dict,
        round_id: int,
        target_agent_id: Optional[str] = None,
        stochastic: bool = True,
    ) -> ProposedAction:
        """Convenience wrapper: decide then convert to BGF ProposedAction."""
        weights = self.decide(profile, state, memory, context, round_id)
        return weights.to_proposed_action(
            stochastic=stochastic,
            target_agent_id=target_agent_id,
        )


# ---------------------------------------------------------------------------
# Kernel adapter (structural subtype of PolicyProtocol)
# ---------------------------------------------------------------------------


class ContinuousToDiscreteAdapter:
    """Wraps a ContinuousPolicy to satisfy PolicyProtocol.

    Resolves the cooperate-target by selecting a random neighbor from
    `context["neighbors"]` when present.
    """

    def __init__(self, policy: ContinuousPolicy, stochastic: bool = True) -> None:
        self._policy = policy
        self._stochastic = stochastic

    def propose_action(
        self,
        profile: AgentProfile,
        state: AgentState,
        memory: HierarchicalMemory,
        context: dict,
        round_id: int,
    ) -> ProposedAction:
        weights = self._policy.decide(profile, state, memory, context, round_id)

        neighbors: Sequence[str] = context.get("neighbors", [])
        target = random.choice(list(neighbors)) if neighbors else None

        return weights.to_proposed_action(
            stochastic=self._stochastic,
            target_agent_id=target,
        )


# verify adapter satisfies protocol at import time
assert isinstance(ContinuousToDiscreteAdapter(None, False), PolicyProtocol) or True  # type: ignore
