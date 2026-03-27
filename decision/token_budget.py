"""Token budget management for LLM prompt construction.

Mistral-7B supports a 2048-token context window. A full BGF prompt
(persona + ESS context + memory + social context + world state) can
approach or exceed this. Silent tokenizer truncation cuts the persona
block first — exactly the grounding we need most.

This module provides a lightweight, dependency-free character-based
estimator (4 chars ≈ 1 token for English text) and helpers to trim
context sections in priority order when the budget is tight.

Priority order (highest → lowest):
  1. System prompt        — never trimmed
  2. State block          — never trimmed (current situation is critical)
  3. Recent memory        — trim window if needed
  4. Persona block        — trim to minimal if necessary
  5. Social/RAG context   — first to drop
  6. Population context   — first to drop
"""

from __future__ import annotations

import warnings

# Conservative estimate: 4 characters per token for English prose.
_CHARS_PER_TOKEN = 4

# Leave 15% headroom for the model's response.
DEFAULT_MAX_TOKENS = 1740  # 2048 * 0.85


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count. Fast, no dependencies."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def fits_in_budget(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> bool:
    return estimate_tokens(text) <= max_tokens


def trim_to_budget(
    system: str,
    persona: str,
    state: str,
    memory: str,
    context: str,
    population_context: str | None,
    social_context: str | None,
    extra: str | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, str | None]:
    """Trim optional context sections to fit within the token budget.

    Returns the same keys with possibly-trimmed values.
    Sections are dropped in reverse-priority order:
      1. extra guidance
      2. social_context (GraphRAG)
      3. population_context (SQL RAG)
      4. memory (window halved, then dropped)
    System, state, and persona are never touched.
    """
    fixed_tokens = (
        estimate_tokens(system)
        + estimate_tokens(persona)
        + estimate_tokens(state)
        + estimate_tokens(context)
        + 50  # formatting overhead
    )
    budget = max_tokens - fixed_tokens

    sections = {
        "memory": memory,
        "population_context": population_context,
        "social_context": social_context,
        "extra": extra,
    }

    # Drop in reverse priority order until we fit
    drop_order = ["extra", "social_context", "population_context"]
    for key in drop_order:
        if budget >= estimate_tokens(sections.get("memory") or ""):
            break
        if sections.get(key):
            budget += estimate_tokens(sections[key] or "")
            sections[key] = None
            if key in ("social_context", "population_context"):
                warnings.warn(
                    f"token_budget: dropped '{key}' to fit within {max_tokens}-token limit. "
                    "RAG context will be absent from this prompt.",
                    stacklevel=3,
                )

    # If memory still too large, halve it (drop older lines)
    if sections["memory"] and estimate_tokens(sections["memory"]) > budget:
        lines = sections["memory"].splitlines()
        sections["memory"] = "\n".join(lines[len(lines) // 2:])

    return {
        "system": system,
        "persona": persona,
        "state": state,
        "context": context,
        "memory": sections["memory"],
        "population_context": sections["population_context"],
        "social_context": sections["social_context"],
        "extra": sections["extra"],
    }
