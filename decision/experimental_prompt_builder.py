from __future__ import annotations

from typing import Optional

from decision.prompt_builder import (
    build_context_block,
    build_memory_block,
    build_persona_block,
    build_state_block,
)

BASE_SYSTEM_PROMPT = """You are a person living in a simulated society. You must decide what action to take this round based on your personal characteristics, your current situation, and the world around you.

You MUST respond with ONLY a JSON object in the following format:
{
  "action_type": "<work|save|cooperate>",
  "target_agent_id": "<neighbor_id or null>",
  "amount": <number>,
  "reasoning_summary": "<brief explanation of your choice>",
  "confidence": <0.0 to 1.0>
}

Valid actions and bounds:
- "work": Earn immediate income. amount must be between 5 and 15.
- "save": Protect stability and reduce strain. amount must be between 5 and 10.
- "cooperate": Help a neighbor. amount must be between 5 and 10. target_agent_id must be one of your neighbors.

Rules:
- Choose exactly one action.
- Keep amount inside the valid bounds for the chosen action.
- If you choose "cooperate", you MUST provide a valid target_agent_id from your neighbors list.
- Respond with ONLY the JSON, no extra text."""

BALANCED_SYSTEM_PROMPT = """You are a living participant in a simulated society. Every round, you must weigh immediate finances, stress regulation, and long-term social capital.

Tradeoffs:
- "work" increases immediate income, but often raises stress.
- "save" preserves stability and can reduce stress, but does not increase income.
- "cooperate" costs immediate resources, but can strengthen trust and social capital.

You MUST respond with ONLY a JSON object in the following format:
{
  "action_type": "<work|save|cooperate>",
  "target_agent_id": "<neighbor_id or null>",
  "amount": <number>,
  "reasoning_summary": "<brief explanation of your choice>",
  "confidence": <0.0 to 1.0>
}

Valid actions and bounds:
- "work": amount must be between 5 and 15.
- "save": amount must be between 5 and 10.
- "cooperate": amount must be between 5 and 10, and target_agent_id must be a valid neighbor.

Rules:
- Choose exactly one action.
- Keep amount inside the valid bounds for the chosen action.
- Explain why that action best fits the current state.
- Do not default to one action without a state-based reason.
- Respond with ONLY the JSON, no extra text."""

SYSTEM_PROMPTS = {
    "base": BASE_SYSTEM_PROMPT,
    "balanced": BALANCED_SYSTEM_PROMPT,
}


def get_system_prompt(mode: str = "balanced") -> str:
    return SYSTEM_PROMPTS.get(mode, BALANCED_SYSTEM_PROMPT)


def build_experimental_prompt(
    profile,
    state,
    memory,
    context: dict,
    round_id: int,
    memory_window: int = 5,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
    use_memory_context: bool = True,
    use_social_context: bool = True,
    use_population_context: bool = True,
    system_prompt_mode: str = "balanced",
    include_balancing_hint: bool = True,
    extra_guidance: Optional[str] = None,
) -> list[dict]:
    persona = build_persona_block(profile)
    state_desc = build_state_block(state)
    memory_desc = build_memory_block(memory, window=memory_window) if use_memory_context else None
    context_desc = build_context_block(context)

    parts = [
        f"Round {round_id}.",
        persona,
        state_desc,
        "Action bounds: work=5-15; save=5-10; cooperate=5-10.",
    ]

    if state.stress >= 0.75:
        parts.append("[WARNING: Your stress is CRITICAL. Choosing work again can be costly unless there is a strong reason.]")

    if use_population_context and population_context:
        parts.append(f"General Population Trends:\n{population_context}")

    if use_social_context and social_context:
        parts.append(f"Social Network Context:\n{social_context}")

    if memory_desc:
        parts.append(memory_desc)

    parts.append(context_desc)

    if extra_guidance:
        parts.append(extra_guidance)

    if include_balancing_hint and system_prompt_mode == "balanced":
        parts.append("HINT: Weigh wealth, stress, and social consequences explicitly; avoid repeating the same action without a concrete reason.")

    parts.append("What action do you take this round? Respond with ONLY the JSON.")
    user_content = "\n\n".join(parts)

    return [
        {"role": "system", "content": get_system_prompt(system_prompt_mode)},
        {"role": "user", "content": user_content},
    ]


def build_experimental_prompt_text(*args, **kwargs) -> str:
    messages = build_experimental_prompt(*args, **kwargs)
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
