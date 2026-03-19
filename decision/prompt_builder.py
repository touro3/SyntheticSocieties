"""
Prompt builder for LLM-based agent decision making.

Constructs structured prompts from agent persona (ESS-derived),
current state, memory, and world context. Designed for Mistral
instruction format.
"""

from __future__ import annotations

import json
from typing import Optional


SYSTEM_PROMPT = """You are a person living in a simulated society. You must decide what action to take this round based on your personal characteristics, your current situation, your memories of past interactions, and the world around you.

You MUST respond with ONLY a JSON object in the following format:
{
  "action_type": "<work|save|cooperate>",
  "target_agent_id": "<neighbor_id or null>",
  "amount": <number>,
  "reasoning_summary": "<brief explanation of your choice>",
  "confidence": <0.0 to 1.0>
}

Valid actions:
- "work": Earn income. amount = how much effort (5-15). No target needed.
- "save": Save wealth for the future. amount = how much to set aside (5-10). No target needed.
- "cooperate": Help a neighbor. amount = resources to share (5-10). target_agent_id = a neighbor's ID.

Rules:
- You MUST choose exactly one action.
- If you choose "cooperate", you MUST specify a target_agent_id from your neighbors list.
- Your reasoning should reflect your personality and situation.
- Respond with ONLY the JSON, no other text."""


def build_persona_block(profile) -> str:
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
        lines.append(f"Politically, you are {pol_str}.")

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


def build_state_block(state) -> str:
    """Build state description from AgentState."""
    return (
        f"Current situation: "
        f"wealth={state.wealth:.1f}, "
        f"stress={state.stress:.2f}, "
        f"satisfaction={state.satisfaction:.2f}."
    )


def build_memory_block(memory, window: int = 5) -> str:
    """Build memory context from recent interactions."""
    if hasattr(memory, "get_recent"):
        recent = memory.get_recent(window)
    else:
        # Fallback if memory is already a list
        recent = memory[-window:] if isinstance(memory, list) else []
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

    neighbors = context.get("network", {}).get("neighbors", [])
    if neighbors:
        lines.append(f"  Your neighbors: {', '.join(neighbors)}")
    else:
        lines.append("  You have no neighbors.")

    return "\n".join(lines)


def build_prompt(
    profile,
    state,
    memory,
    context: dict,
    round_id: int,
    memory_window: int = 5,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
) -> list[dict]:
    """
    Build a complete chat-format prompt for the LLM.

    Returns a list of message dicts compatible with transformers
    chat template format: [{"role": "...", "content": "..."}].
    """
    persona = build_persona_block(profile)
    state_desc = build_state_block(state)
    memory_desc = build_memory_block(memory, window=memory_window)
    context_desc = build_context_block(context)

    user_content = f"Round {round_id}.\n\n{persona}\n\n{state_desc}\n\n"
    
    if population_context:
        user_content += f"General Population Trends:\n{population_context}\n\n"
        
    if social_context:
        user_content += f"Social Network Context:\n{social_context}\n\n"

    user_content += f"{memory_desc}\n\n{context_desc}\n\n"
    user_content += "What action do you take this round? Respond with ONLY the JSON."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return messages


def build_prompt_text(
    profile,
    state,
    memory,
    context: dict,
    round_id: int,
    memory_window: int = 5,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
) -> str:
    """
    Build a plain-text version of the prompt (for logging/debugging).
    """
    messages = build_prompt(
        profile, state, memory, context, round_id, memory_window,
        social_context=social_context, population_context=population_context
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
