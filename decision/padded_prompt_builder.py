"""Length-controlled padded prompt builder for causal ablation.

Causal inference and ablation formalization.

When Condition B (grounded) adds ESS persona + RAG context, it also adds
tokens. A skeptic could argue the LLM behaves differently because of
prompt length, not semantic content. This module builds prompts padded to
the same token count as grounded prompts, but with semantically empty
filler. If the grounding effect survives vs the padded control, it is the
content, not the length, that matters.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from decision.prompt_builder import (
    build_context_block,
    build_memory_block,
    build_state_block,
)
from decision.system_prompts import BASE_SYSTEM_PROMPT
from decision.token_budget import estimate_tokens

# ── Semantically empty filler pool ───────────────────────────────────────
# These sentences discuss generic, non-specific concepts. They must NOT
# contain any ESS-specific data, demographic grounding, or culturally
# loaded statements that could influence agent decision-making.

_PADDING_POOL: list[str] = [
    "The simulation environment contains multiple interacting entities.",
    "Economic decisions involve trade-offs between competing objectives.",
    "Social networks can exhibit various structural properties.",
    "Resource allocation is a fundamental challenge in multi-agent systems.",
    "Cooperative behavior can emerge from individual incentives.",
    "Environmental factors may influence decision-making processes.",
    "Agents observe outcomes and adjust their strategies accordingly.",
    "The passage of time introduces new opportunities and constraints.",
    "Information asymmetry affects the quality of decisions made.",
    "Collective outcomes emerge from individual-level interactions.",
    "Sequential decision-making requires balancing short and long-term goals.",
    "Network position can affect access to resources and information.",
    "Feedback loops between agents create complex dynamic patterns.",
    "Bounded rationality limits the optimality of real-world decisions.",
    "Heterogeneity among agents produces diverse behavioral outcomes.",
    "Repeated interactions allow learning and adaptation over time.",
    "Institutional rules shape the incentive landscape for participants.",
    "Coordination problems arise when agents have partially aligned goals.",
    "Stochastic elements introduce variability into deterministic processes.",
    "Emergent properties arise from simple rules applied at scale.",
    "The structure of interaction networks shapes aggregate dynamics.",
    "Temporal discounting affects how agents value future outcomes.",
    "Multi-objective optimization requires balancing competing criteria.",
    "Path dependence means early decisions constrain later options.",
    "Scale effects can amplify or dampen individual-level phenomena.",
    "Adaptive systems respond to environmental changes over time.",
    "Strategic interdependence characterizes multi-agent environments.",
    "Information propagation follows network connectivity patterns.",
    "Equilibrium analysis identifies stable states of dynamic systems.",
    "Perturbation analysis reveals system sensitivity to parameter changes.",
]


# ── Padded prompt builder ────────────────────────────────────────────────


def build_padded_prompt(
    profile: Any,
    state: Any,
    memory: Any,
    context: dict,
    round_id: int,
    target_token_count: int,
    seed: int = 42,
    memory_window: int = 5,
) -> list[dict]:
    """Build a prompt with same token count as grounded but without ESS content.

    Uses a minimal (no-persona) base prompt, then pads with filler
    sentences until the user message reaches ``target_token_count``
    tokens (within a tolerance of +-25).

    Args:
        profile: AgentProfile (persona not injected — replaced by anonymous).
        state: AgentState.
        memory: HierarchicalMemory.
        context: World context dict.
        round_id: Current simulation round.
        target_token_count: Desired token count for the user message.
        seed: Random seed for filler selection.
        memory_window: Memory window size.

    Returns:
        Two-message list: [system, user].
    """
    system = BASE_SYSTEM_PROMPT

    # Build base prompt without persona or RAG
    persona = "You are a participant in an economic simulation."
    state_desc = build_state_block(state)
    memory_desc = build_memory_block(memory, window=memory_window)
    context_desc = build_context_block(context)

    parts = [f"Round {round_id}.", persona, state_desc]
    if memory_desc:
        parts.append(memory_desc)
    parts.append(context_desc)
    parts.append("What action do you take this round? Respond with ONLY the JSON.")

    user_content = "\n\n".join(parts)

    # Pad to target token count
    current_tokens = estimate_tokens(user_content)
    if current_tokens < target_token_count:
        rng = random.Random(seed)
        padding_lines: list[str] = []
        pool = list(_PADDING_POOL)

        # Guard: cap iterations to prevent infinite loop if estimate_tokens
        # is miscalibrated (e.g. returns 0 due to a broken tokenizer).
        max_pad_iters = len(_PADDING_POOL) * 5
        pad_iter = 0
        while current_tokens < target_token_count - 25 and pad_iter < max_pad_iters:
            pad_iter += 1
            if not pool:
                pool = list(_PADDING_POOL)  # Recycle if exhausted
            line = rng.choice(pool)
            pool.remove(line)
            padding_lines.append(line)
            candidate = user_content + "\n\n" + "\n".join(padding_lines)
            current_tokens = estimate_tokens(candidate)

        if padding_lines:
            user_content = user_content + "\n\n" + "\n".join(padding_lines)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def measure_grounded_token_count(
    profile: Any,
    state: Any,
    memory: Any,
    context: dict,
    round_id: int,
    social_context: Optional[str] = None,
    population_context: Optional[str] = None,
    memory_window: int = 5,
) -> int:
    """Measure the token count of a fully grounded prompt's user message.

    This is the target token count that ``build_padded_prompt`` should
    match for a fair length-controlled comparison.
    """
    from decision.prompt_builder import build_persona_block

    parts = [
        f"Round {round_id}.",
        build_persona_block(profile),
        build_state_block(state),
    ]

    memory_desc = build_memory_block(memory, window=memory_window)
    if memory_desc:
        parts.append(memory_desc)

    parts.append(build_context_block(context))

    if population_context:
        parts.append(population_context)
    if social_context:
        parts.append(social_context)

    parts.append("What action do you take this round? Respond with ONLY the JSON.")

    user_content = "\n\n".join(parts)
    return estimate_tokens(user_content)
