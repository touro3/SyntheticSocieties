from __future__ import annotations

from typing import Optional

from decision.prompt_builder import (
    build_context_block,
    build_memory_block,
    build_persona_block,
    build_state_block,
)
from decision.system_prompts import (
    EXPERIMENTAL_BALANCED_SYSTEM_PROMPT as BALANCED_SYSTEM_PROMPT,
)
from decision.system_prompts import (
    EXPERIMENTAL_BASE_SYSTEM_PROMPT as BASE_SYSTEM_PROMPT,
)
from decision.token_budget import DEFAULT_MAX_TOKENS, trim_to_budget

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
    max_tokens: Optional[int] = None,
) -> list[dict]:
    system_text = get_system_prompt(system_prompt_mode)
    persona = build_persona_block(profile)
    state_desc = build_state_block(state)
    memory_desc = build_memory_block(memory, window=memory_window) if use_memory_context else ""
    context_desc = build_context_block(context)

    pop_ctx = population_context if use_population_context else None
    social_ctx = social_context if use_social_context else None

    # Trim optional sections to stay within the model's context window.
    budget = max_tokens or DEFAULT_MAX_TOKENS
    trimmed = trim_to_budget(
        system=system_text,
        persona=persona,
        state=state_desc,
        memory=memory_desc,
        context=context_desc,
        population_context=pop_ctx,
        social_context=social_ctx,
        extra=extra_guidance,
        max_tokens=budget,
    )

    parts = [
        f"Round {round_id}.",
        trimmed["persona"],
        trimmed["state"],
        "Action bounds: work=5-15; save=5-10; cooperate=5-10.",
    ]

    if state.stress >= 0.75:
        parts.append(f"[Your stress level is critically high ({state.stress:.2f}).]" )

    if trimmed["population_context"]:
        parts.append(f"General Population Trends:\n{trimmed['population_context']}")

    if trimmed["social_context"]:
        parts.append(f"Social Network Context:\n{trimmed['social_context']}")

    if trimmed["memory"]:
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


def build_experimental_prompt_text(*args, **kwargs) -> str:
    messages = build_experimental_prompt(*args, **kwargs)
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)

