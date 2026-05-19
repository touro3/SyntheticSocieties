"""
Prompt builder for LLM-based agent decision making.

Constructs structured prompts from agent persona (ESS-derived),
current state, memory, and world context. Designed for Mistral
instruction format.

Key entry point for most callers:
    messages = build_prompt(profile, state, memory, context, round_id)

The ``AblationLevel`` class documents exactly what each grounding level adds.
"""

from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

from decision.constants import STRESS_CRITICAL
from decision.system_prompts import (
    BALANCED_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    get_shuffled_system_prompt,
)
from decision.token_budget import DEFAULT_MAX_TOKENS, trim_to_budget

if TYPE_CHECKING:
    from agents.memory import HierarchicalMemory
    from agents.profile import AgentProfile
    from agents.state import AgentState


class AblationLevel(IntEnum):
    """BGF prompt ablation levels — each adds one grounding component.

    Used throughout build_prompt() and build_state_block() to control
    how much ESS-derived context is injected into each LLM prompt.

        BASELINE       (0) — plain prompt; no grounding
        STRESS_AWARE   (1) — + explicit stress-salience warning
        COOPERATION    (2) — + cooperation-incentive hint
        TRUST_SURFACED (3) — + trust-network dict in the state block
        BALANCED       (4) — + balanced system-prompt phrasing
        FULL           (5) — all components (default for Condition B)

    Because IntEnum inherits from int, existing ``ablation_level >= N``
    comparisons continue to work unchanged.
    """

    BASELINE = 0  # No grounding — equivalent to Condition A
    STRESS_AWARE = 1  # Adds: stress salience warning
    COOPERATION = 2  # Adds: cooperation-incentive hint
    TRUST_SURFACED = 3  # Adds: trust network surfaced in state
    BALANCED = 4  # Adds: balanced system-prompt phrasing
    FULL = 5  # Full grounding — default for Condition B


def get_neighbors(context: dict) -> list[str]:
    """Extract the list of neighbor agent IDs from a world-context dict.

    All policies need this one-liner; centralising it avoids silent
    divergence if the context schema ever changes.

    Args:
        context: World-context dict produced by World.get_agent_context().

    Returns:
        List of neighbor agent IDs (may be empty).
    """
    return context.get("network", {}).get("neighbors", [])


def build_persona_block(profile: AgentProfile) -> str:
    """Build a natural-language persona description from AgentProfile."""
    lines = [f"You are {profile.agent_id}, age {profile.age}."]

    if profile.gender is not None:
        gender_str = "male" if profile.gender == 1 else "female" if profile.gender == 2 else "unknown"
        lines.append(f"Gender: {gender_str}.")

    if profile.country:
        lines.append(f"Country: {profile.country}.")

    if profile.education:
        lines.append(f"Education: {profile.education}.")

    if profile.occupation:
        lines.append(f"Occupation: {profile.occupation}.")

    if profile.location:
        lines.append(f"Location type: {profile.location}.")

    # Trust & social attributes
    trust_parts = []
    if profile.trust_people is not None:
        level = _level_word(profile.trust_people)
        trust_parts.append(f"You have {level} trust in other people ({profile.trust_people:.2f}/1.0)")
    if profile.trust_institutions is not None:
        level = _level_word(profile.trust_institutions)
        trust_parts.append(f"{level} trust in institutions ({profile.trust_institutions:.2f}/1.0)")
    if trust_parts:
        lines.append(". ".join(trust_parts) + ".")

    # Political & values
    if profile.political_orientation is not None:
        pol = profile.political_orientation
        if pol < 0.3:
            pol_str = "left-leaning"
        elif pol < 0.45:
            pol_str = "center-left"
        elif pol < 0.55:
            pol_str = "centrist"
        elif pol < 0.7:
            pol_str = "center-right"
        else:
            pol_str = "right-leaning"
        lines.append(f"Politically, you are {pol_str} ({pol:.2f}/1.0).")

    if profile.life_satisfaction is not None:
        level = _level_word(profile.life_satisfaction)
        lines.append(f"Your life satisfaction is {level} ({profile.life_satisfaction:.2f}/1.0).")

    if profile.happiness is not None:
        level = _level_word(profile.happiness)
        lines.append(f"Your happiness is {level} ({profile.happiness:.2f}/1.0).")

    # Personality traits
    if profile.risk_tolerance is not None:
        level = _level_word(profile.risk_tolerance)
        lines.append(f"Your risk tolerance is {level} ({profile.risk_tolerance:.2f}/1.0).")

    if profile.competitiveness is not None:
        level = _level_word(profile.competitiveness)
        lines.append(f"Your competitiveness is {level} ({profile.competitiveness:.2f}/1.0).")

    if profile.social_activity is not None:
        level = _level_word(profile.social_activity)
        lines.append(f"Your social activity level is {level} ({profile.social_activity:.2f}/1.0).")

    if profile.religiosity is not None:
        lines.append(f"You are {'religious' if profile.religiosity > 0.5 else 'not religious'}.")

    return " ".join(lines)


def build_state_block(state: AgentState, ablation_level: int = 5) -> str:
    """Build state description from AgentState, subject to ablation testing."""
    base = (
        f"Current situation: "
        f"wealth={state.wealth:.1f}, "
        f"stress={state.stress:.2f}, "
        f"satisfaction={state.satisfaction:.2f}."
    )

    # Neutral stress observation (no action recommendations)
    if ablation_level >= AblationLevel.STRESS_AWARE and state.stress >= STRESS_CRITICAL:
        base += f"\n[Your stress level is critically high ({state.stress:.2f}).]"

    # V3: Surface Agent Trust Dictionary directly in state
    if ablation_level >= AblationLevel.TRUST_SURFACED and hasattr(state, "trust_network") and state.trust_network:
        # Only show neighbors where trust > 0
        active_trust = dict(
            sorted(
                ((k, round(v, 2)) for k, v in state.trust_network.items() if v > 0),
                key=lambda kv: kv[0],
            )
        )
        if active_trust:
            base += f"\n[Your internal trust toward specific neighbors based on their past help: {active_trust}]"

    return base


def build_memory_block(
    memory: HierarchicalMemory,
    window: int = 5,
    profile: Optional[AgentProfile] = None,
    level=None,
) -> str:
    """Build memory context from recent interactions, subject to ablation level.

    The ``level`` parameter controls how much memory is surfaced to the LLM:

      M0 — no memory context at all
      M1 — sliding window of recent events only (no reflection, no archive)
      M2 — recent events + archive count (no reflection text)
      M3 — full hierarchical: reflection + important recent + drift anchor [default]

    If ``level`` is None, the value is read from ``memory.level`` (defaulting
    to M3 for backwards compatibility with existing HierarchicalMemory objects).

    When ``profile`` is provided (M3 only), a re-anchoring cue is injected
    when the agent's cooperation rate has drifted from ESS-derived expectations.
    """
    from agents.memory import MemoryLevel  # local import to avoid circular dep

    # Resolve effective level: explicit arg beats memory attribute.
    if level is None:
        level = getattr(memory, "level", MemoryLevel.M3)
    effective = MemoryLevel(int(level))

    # ── M0: no memory ─────────────────────────────────────────────────────────
    if effective == MemoryLevel.M0:
        return "No memory context available."

    # ── M1: window only ───────────────────────────────────────────────────────
    if effective == MemoryLevel.M1:
        recent = memory.get_recent(window) if hasattr(memory, "get_recent") else []
        if not recent:
            return "You have no memories of past interactions yet."
        lines = ["Your recent memories:"]
        for item in recent:
            line = f"  Round {item.round_id}: you chose '{item.event_type}'"
            if item.partner_id:
                line += f" with {item.partner_id}"
            if item.content:
                line += f" ({item.content})"
            lines.append(line)
        return "\n".join(lines)

    # ── M2: window + archive count, no reflections ────────────────────────────
    if effective == MemoryLevel.M2:
        recent = memory.get_recent(window) if hasattr(memory, "get_recent") else []
        lines = []
        archive_count = len(getattr(memory, "archive", []))
        if archive_count > 0:
            lines.append(f"[Memory archive: {archive_count} older events stored, not shown]")
        if not recent:
            if not lines:
                return "You have no memories of past interactions yet."
            return "\n".join(lines)
        lines.append("Your recent memories:")
        for item in recent:
            line = f"  Round {item.round_id}: you chose '{item.event_type}'"
            if item.partner_id:
                line += f" with {item.partner_id}"
            if item.content:
                line += f" ({item.content})"
            lines.append(line)
        return "\n".join(lines)

    # ── M3: full hierarchical (original behaviour) ────────────────────────────
    # Use importance-scored retrieval when available.
    if hasattr(memory, "get_important_recent"):
        recent = memory.get_important_recent(window)
    elif hasattr(memory, "get_recent"):
        recent = memory.get_recent(window)
    else:
        recent = memory[-window:] if isinstance(memory, list) else []

    lines = []

    # Reflection: compressed summary of full history (recent + archive).
    if hasattr(memory, "generate_reflection"):
        reflection = memory.generate_reflection()
        if reflection:
            lines.append(f"[Memory summary] {reflection}")

    # NOTE: persona re-anchoring is intentionally NOT applied here.
    # Injecting an expected cooperation rate derived from trust/risk attributes
    # into LLM prompts during a simulation run conflates the grounding mechanism
    # under study with the measurement of its effect, confounding Condition A vs B.
    # See _build_persona_anchor() below for the analysis-only utility.

    if not recent:
        if not lines:
            return "You have no memories of past interactions yet."
        return "\n".join(lines)

    lines.append("Your recent memories:")
    for item in recent:
        line = f"  Round {item.round_id}: you chose '{item.event_type}'"
        if item.partner_id:
            line += f" with {item.partner_id}"
        if item.content:
            line += f" ({item.content})"
        lines.append(line)

    return "\n".join(lines)


def build_collective_context_block(collective_context: Optional[list[str]] = None) -> str:
    """Build shared world-fact context visible to all agents."""
    if not collective_context:
        return ""
    lines = ["Community knowledge:"]
    for item in collective_context:
        text = str(item).strip()
        if text:
            lines.append(f"  {text}")
    return "\n".join(lines)


# Drift detection threshold: if |actual - expected| > this, produce an anchor.
_DRIFT_THRESHOLD = 0.25


def _build_persona_anchor(
    memory: HierarchicalMemory,
    profile: AgentProfile,
) -> Optional[str]:
    """Generate a persona re-anchoring string for POST-HOC ANALYSIS ONLY.

    WARNING: This function MUST NOT be called from build_memory_block() or any
    code path that constructs prompts used during a live simulation run.
    Injecting a cooperation-rate expectation derived from trust/risk attributes
    directly into agent prompts is a confound: it manipulates agent behavior in
    the grounded condition (Condition B) via a formula that is NOT derived from
    ESS data, violating the empirical grounding claim.

    The formula `expected_coop = 0.2 + 0.6 * trust * (1.0 - risk)` below is a
    heuristic approximation. It is retained here only so that analysis scripts
    and tests that study drift can compute a drift signal, NOT so that the LLM
    is told what cooperation rate to target.

    Returns None if no drift is detected or insufficient data.
    """
    if not hasattr(memory, "get_action_distribution"):
        return None

    dist = memory.get_action_distribution(weighted=True)
    if not dist:
        return None

    # Need enough history to detect drift (at least 5 events).
    all_items = memory.archive + getattr(memory, "_effective_recent", memory.recent)
    if len(all_items) < 5:
        return None

    actual_coop = dist.get("cooperate", 0.0)

    # Derive expected cooperation from persona attributes.
    trust = getattr(profile, "trust_people", None) or 0.5
    risk = getattr(profile, "risk_tolerance", None) or 0.5
    expected_coop = 0.2 + 0.6 * trust * (1.0 - risk)

    drift = actual_coop - expected_coop

    if abs(drift) <= _DRIFT_THRESHOLD:
        return None

    # Build a non-directive grounding cue based on persona.
    trust_word = _level_word(trust)
    if drift < 0:
        # Under-cooperating relative to persona.
        return (
            f"[Persona reminder] Your core disposition: you have {trust_word} "
            f"trust in others ({trust:.2f}/1.0). People with your profile "
            f"tend to cooperate about {expected_coop:.0%} of the time. "
            f"Consider whether your recent choices still reflect who you are."
        )
    else:
        # Over-cooperating relative to persona.
        return (
            f"[Persona reminder] Your core disposition: you have {trust_word} "
            f"trust in others ({trust:.2f}/1.0). People with your profile "
            f"tend to cooperate about {expected_coop:.0%} of the time. "
            f"Consider whether your recent generosity still reflects your "
            f"actual comfort level with risk."
        )


def build_context_block(context: dict) -> str:
    """Build world context description."""
    lines = ["World state:"]

    world = context.get("world", {})
    if "prices" in world:
        prices = world["prices"]
        lines.append(f"  Food price: {prices.get('food', 'unknown')}")

    if "public_signal" in world:
        signal = world["public_signal"]
        economy = signal.get("economy", "unknown")
        lines.append(f"  Economy: {economy}")

    if "resources" in world:
        resources = world["resources"]
        jobs = resources.get("jobs", "unknown")
        lines.append(f"  Available jobs: {jobs}")

    neighbors = get_neighbors(context)
    if neighbors:
        lines.append(f"  Your neighbors: {', '.join(neighbors)}")
    else:
        lines.append("  You have no neighbors.")

    return "\n".join(lines)


def build_prompt(
    profile: AgentProfile,
    state: AgentState,
    memory: HierarchicalMemory,
    context: dict,
    round_id: int,
    memory_window: int = 5,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
    ablation_level: int = 5,
    max_tokens: Optional[int] = None,
    collective_context: Optional[list[str]] = None,
) -> list[dict]:
    """
    Build a complete chat-format prompt for the LLM based on ablation level.
    V0: Baseline prompt
    V1: Add explicit stress salience penalties
    V2: Add explicit cooperation incentives
    V3: Surface trust/memory in state
    V4: Re-balance generic action phrasing
    """
    persona = build_persona_block(profile)
    state_desc = build_state_block(state, ablation_level)
    memory_desc = build_memory_block(memory, window=memory_window, profile=profile)
    collective_desc = build_collective_context_block(collective_context)
    memory_with_collective = f"{collective_desc}\n\n{memory_desc}" if collective_desc else memory_desc
    context_desc = build_context_block(context)
    # Pin the action-order shuffle to (round_id, agent_id) so that any
    # subsequent call with the same inputs (e.g. the logging path) produces
    # the identical system prompt — without this the logged prompt diverges
    # from what the model actually received, corrupting the prompt log.
    # Uses SHA-256 (not Python's built-in hash()) for cross-process stability:
    # hash() is randomized per-interpreter since Python 3.3 (PYTHONHASHSEED),
    # which would produce different action orderings across machines/processes.
    shuffle_seed = int(hashlib.sha256(f"{round_id}:{profile.agent_id}".encode()).hexdigest()[:8], 16)
    system_text = get_shuffled_system_prompt(seed=shuffle_seed)

    # No extra guidance — prompt must not bias toward specific actions.
    extra = None

    # Trim optional sections to stay within the model's context window.
    trimmed = trim_to_budget(
        system=system_text,
        persona=persona,
        state=state_desc,
        memory=memory_with_collective,
        context=context_desc,
        population_context=population_context,
        social_context=social_context,
        extra=extra,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
    )

    parts = [f"Round {round_id}.", trimmed["persona"], trimmed["state"]]

    if trimmed["population_context"]:
        parts.append(f"General Population Trends:\n{trimmed['population_context']}")

    if trimmed["social_context"]:
        parts.append(f"Social Network Context:\n{trimmed['social_context']}")

    parts.append(trimmed["memory"])
    parts.append(trimmed["context"])

    if trimmed["extra"]:
        parts.append(trimmed["extra"])

    parts.append("What action do you take this round? Respond with ONLY the JSON.")

    user_content = "\n\n".join(p for p in parts if p)

    return [
        {"role": "system", "content": trimmed["system"]},
        {"role": "user", "content": user_content},
    ]


def build_prompt_staged(
    profile: AgentProfile,
    state: AgentState,
    memory: HierarchicalMemory,
    context: dict,
    round_id: int,
    memory_window: int = 5,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
    ablation_level: int = 5,
    max_tokens: Optional[int] = None,
    collective_context: Optional[list[str]] = None,
) -> list[dict]:
    """Build a chat-format prompt via 4 sequential construction stages.

    Mirrors MiroFish's multi-stage LLM config generation strategy: instead
    of assembling one massive prompt string, each stage adds a focused block.
    This prevents any single component from crowding out others when the
    token budget is tight and makes budget allocation explicit.

    Stage 1 — Identity:    system prompt + persona (who the agent is)
    Stage 2 — State:       current wealth/stress/trust (what's happening now)
    Stage 3 — History:     memory reflection + recent events (what happened)
    Stage 4 — World/RAG:   environment context + population/social signals

    Each stage is trimmed independently before assembly so that history never
    cannibalises persona or vice-versa. The final action instruction is always
    appended last and is never trimmed.
    """
    # Shuffle action ordering in the system prompt with the same SHA-256
    # per-(round, agent) seed used by build_prompt(), so the staged builder
    # does not reintroduce LLM position bias by always listing actions in a
    # fixed order. Deterministic: identical seeds → identical ordering.
    shuffle_seed = int(hashlib.sha256(f"{round_id}:{profile.agent_id}".encode()).hexdigest()[:8], 16)
    system_text = get_shuffled_system_prompt(seed=shuffle_seed)

    # ── Stage 1: Identity ────────────────────────────────────────────────────
    persona = build_persona_block(profile)

    # ── Stage 2: State ───────────────────────────────────────────────────────
    state_desc = build_state_block(state, ablation_level)

    # ── Stage 3: History ─────────────────────────────────────────────────────
    memory_desc = build_memory_block(memory, window=memory_window, profile=profile)
    collective_desc = build_collective_context_block(collective_context)
    memory_with_collective = f"{collective_desc}\n\n{memory_desc}" if collective_desc else memory_desc

    # ── Stage 4: World + RAG context ─────────────────────────────────────────
    context_desc = build_context_block(context)

    # Trim using the same priority-aware budget manager as build_prompt().
    # The previous per-stage character heuristic (~4 chars/token) was both
    # inconsistent with build_prompt() and less accurate than the tokenizer-
    # backed trim_to_budget() call below.
    budget = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

    trimmed = trim_to_budget(
        system=system_text,
        persona=persona,
        state=state_desc,
        memory=memory_with_collective,
        context=context_desc,
        population_context=population_context,
        social_context=social_context,
        extra=None,
        max_tokens=budget,
    )

    persona_t = trimmed["persona"]
    state_t = trimmed["state"]
    memory_t = trimmed["memory"]
    context_t = trimmed["context"]
    pop_t = trimmed["population_context"]
    social_t = trimmed["social_context"]

    # ── Assemble ──────────────────────────────────────────────────────────────
    parts = [f"Round {round_id}.", persona_t, state_t]

    if pop_t:
        parts.append(f"General Population Trends:\n{pop_t}")
    if social_t:
        parts.append(f"Social Network Context:\n{social_t}")

    parts.append(memory_t)
    parts.append(context_t)
    parts.append("What action do you take this round? Respond with ONLY the JSON.")

    user_content = "\n\n".join(p for p in parts if p)

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]


def build_prompt_text(
    profile,
    state,
    memory,
    context: dict,
    round_id: int,
    memory_window: int = 5,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
    ablation_level: int = 5,
    max_tokens: Optional[int] = None,
    collective_context: Optional[list[str]] = None,
) -> str:
    """
    Build a plain-text version of the prompt (for logging/debugging).
    """
    messages = build_prompt(
        profile,
        state,
        memory,
        context,
        round_id,
        memory_window,
        social_context=social_context,
        population_context=population_context,
        ablation_level=ablation_level,
        max_tokens=max_tokens,
        collective_context=collective_context,
    )
    parts = []
    for msg in messages:
        parts.append(f"[{msg['role'].upper()}]\n{msg['content']}")
    return "\n\n".join(parts)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _level_word(value: float) -> str:
    """Convert a [0, 1] score to a descriptive word."""
    if value < 0.2:
        return "very low"
    if value < 0.4:
        return "low"
    if value < 0.6:
        return "moderate"
    if value < 0.8:
        return "high"
    return "very high"
