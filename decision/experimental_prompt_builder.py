from __future__ import annotations

from typing import Optional

from decision.prompt_builder import (
    build_context_block,
    build_memory_block,
    build_persona_block,
    build_state_block,
)
from decision.system_prompts import (
    EXPERIMENTAL_BASE_SYSTEM_PROMPT as BASE_SYSTEM_PROMPT,
    EXPERIMENTAL_BALANCED_SYSTEM_PROMPT as BALANCED_SYSTEM_PROMPT,
)

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
