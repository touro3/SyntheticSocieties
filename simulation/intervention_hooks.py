"""Developer intervention hooks for causal diagnostics (Phase 3, audit response).

Research motivation
-------------------
To show agents react *causally* (not myopically) we must be able to inject
controlled shocks and measure directional responses against social-science
literature (e.g. a trust shock should depress cooperation; a scarcity shock
should shift effort/saving). This module provides a small, declarative
registry of timed interventions plus a memory-counterfactual helper.

Design
------
Mirrors :class:`environment.policy_intervention.InterventionEngine` (a
``apply(round_id, agents, ...)`` scheduler returning event dicts) and rides
the **existing** ``world.state.pending_injections`` queue for world-level
shocks — the kernel core loop is untouched. Agent-level shocks (trust) are
applied directly to ``agent.state.trust`` because the world layer has no
agent handle.

Use it from an experiment/diagnostic script: instantiate the registry, then
call ``apply(round_id, agents, world)`` once per round (or just at the
trigger round). Nothing here is wired into the production kernel by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_VALID_KINDS = ("trust_shock", "scarcity", "wealth_shock")


@dataclass
class InterventionHook:
    """One declarative, timed intervention.

    Attributes:
        trigger_round: Round at which the hook fires.
        kind: One of ``trust_shock`` | ``scarcity`` | ``wealth_shock``.
        params: Kind-specific parameters (e.g. ``{"magnitude": 0.5}``;
            ``trust_shock`` also accepts ``{"partner": "agent_3"}``).
        repeat: If True, fire every round from ``trigger_round`` onward.
        label: Human-readable name for logs/thesis methodology.
    """

    trigger_round: int
    kind: str
    params: dict = field(default_factory=dict)
    repeat: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Unknown intervention kind {self.kind!r}. Valid: {_VALID_KINDS}")

    def due(self, round_id: int) -> bool:
        return round_id == self.trigger_round or (self.repeat and round_id >= self.trigger_round)


def _apply_trust_shock(agents, params: dict) -> dict:
    """Depress per-partner trust. ``magnitude`` ∈ [0,1] is the fractional drop.

    Optional ``partner`` restricts the shock to trust toward that agent id.
    Returns an event dict for logging the directional intervention.
    """
    magnitude = float(params.get("magnitude", 0.5))
    partner = params.get("partner")
    affected = 0
    total_delta = 0.0
    for ag in agents:
        trust = getattr(ag.state, "trust", None)
        if not isinstance(trust, dict):
            continue
        keys = [partner] if partner is not None else list(trust.keys())
        for k in keys:
            if k in trust:
                before = trust[k]
                trust[k] = max(0.0, before * (1.0 - magnitude))
                total_delta += trust[k] - before
                affected += 1
    return {
        "kind": "trust_shock",
        "magnitude": magnitude,
        "partner": partner,
        "n_trust_edges_affected": affected,
        "total_trust_delta": round(total_delta, 4),
    }


def _queue_world_injection(world, event_type: str, payload: dict) -> dict:
    """Append a world-level shock to the existing pending_injections queue."""
    injection = {"event_type": event_type, "payload": payload}
    world.state.pending_injections.append(injection)
    return {"kind": event_type, "queued": True, "payload": payload}


class InterventionHookRegistry:
    """Schedules :class:`InterventionHook` s and applies the due ones."""

    def __init__(self, hooks: list[InterventionHook] | None = None) -> None:
        self.hooks = list(hooks or [])

    def add(self, hook: InterventionHook) -> None:
        self.hooks.append(hook)

    def apply(self, round_id: int, agents, world=None) -> list[dict]:
        """Fire every hook due this round; return event dicts for logging.

        ``trust_shock`` mutates ``agent.state.trust`` immediately.
        ``scarcity`` / ``wealth_shock`` are queued onto
        ``world.state.pending_injections`` so the unmodified
        ``World.apply_exogenous_updates`` applies them next tick.
        """
        events: list[dict] = []
        for hook in self.hooks:
            if not hook.due(round_id):
                continue
            if hook.kind == "trust_shock":
                ev = _apply_trust_shock(agents, hook.params)
            elif hook.kind in ("scarcity", "wealth_shock"):
                if world is None:
                    raise ValueError(f"{hook.kind} requires a world to queue the injection")
                ev = _queue_world_injection(world, hook.kind, dict(hook.params))
            else:  # pragma: no cover — guarded by __post_init__
                continue
            ev.update({"round_id": round_id, "label": hook.label})
            logger.info("Intervention fired: %s @ round %d (%s)", hook.kind, round_id, hook.label)
            events.append(ev)
        return events


# ── Memory counterfactual helper ──────────────────────────────────────────


def delete_betrayal_memories(agent, partner_id: str | None = None) -> int:
    """Excise betrayal memories from an agent (memory-deletion ablation).

    A *betrayal* is a ``cooperate`` action whose recorded outcome was not
    reciprocated (``outcome["reciprocated"] is False``). Removing it lets us
    measure whether the agent's subsequent policy toward ``partner_id``
    changes vs a control agent that keeps the memory — direct evidence the
    memory is behaviorally load-bearing.

    Operates on both the live ``recent`` list and ``archive`` so the belief
    cannot leak back via reflection. Returns the number of items removed.
    """

    def _is_betrayal(item) -> bool:
        if item.event_type != "cooperate":
            return False
        if partner_id is not None and item.partner_id != partner_id:
            return False
        outcome = item.outcome or {}
        return outcome.get("reciprocated") is False

    removed = 0
    for attr in ("recent", "archive"):
        bucket = getattr(agent.memory, attr, None)
        if not isinstance(bucket, list):
            continue
        kept = [it for it in bucket if not _is_betrayal(it)]
        removed += len(bucket) - len(kept)
        setattr(agent.memory, attr, kept)

    # Invalidate any cached reflection so the wiped event cannot resurface.
    if hasattr(agent.memory, "_cache_dirty"):
        agent.memory._cache_dirty = True
        agent.memory._reflection_cache = None
    return removed
