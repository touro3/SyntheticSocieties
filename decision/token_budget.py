"""Token budget management for LLM prompt construction.

A full BGF prompt (persona + ESS context + memory + social context + world
state) can approach or exceed the model's context window. Silent tokenizer
truncation cuts the persona block first — exactly the grounding we need most.

This module provides a lightweight, dependency-free character-based estimator
(4 chars ≈ 1 token for English text) and helpers to trim context sections in
priority order when the budget is tight.

Priority order (highest → lowest):
  1. System prompt        — never trimmed
  2. State block          — never trimmed (current situation is critical)
  3. Recent memory        — trim window if needed
  4. Persona block        — trim to minimal if necessary
  5. Social/RAG context   — first to drop
  6. Population context   — first to drop

Per-model prompt budgets (prompt tokens only; excludes response headroom):
  - Mistral-7B-Instruct-v0.3  (32k ctx):  4096  — quality sweet spot for 7B attention
  - Qwen2.5-7B-Instruct       (131k ctx): 6144  — handles long context well
  - GPT-4o-mini               (128k ctx): 8192  — cost bounded by small agent count
  - default                               3072  — safe floor for unknown models
"""

from __future__ import annotations

import warnings

# Conservative estimate: 4 characters per token for English prose.
_CHARS_PER_TOKEN = 4

# Default budget for unknown/unconfigured models.
DEFAULT_MAX_TOKENS = 3072

# Per-model prompt budgets keyed by model_id substring.
_MODEL_BUDGETS: dict[str, int] = {
    "mistral-7b": 4096,
    "mistral-7b-instruct": 4096,
    "qwen2.5-7b": 6144,
    "gpt-4o-mini": 8192,
}


def budget_for_model(model_id: str) -> int:
    """Return the prompt-token budget for the given model_id.

    Matches case-insensitively against known model substrings.  Falls back to
    DEFAULT_MAX_TOKENS for unrecognised models.
    """
    lower = model_id.lower()
    for key, budget in _MODEL_BUDGETS.items():
        if key in lower:
            return budget
    return DEFAULT_MAX_TOKENS


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

    # If memory still too large, halve repeatedly until it fits.
    # Hard floor of 4 lines preserves at least the reflection + most recent event.
    if sections["memory"] and estimate_tokens(sections["memory"]) > budget:
        lines = sections["memory"].splitlines()
        while lines and estimate_tokens("\n".join(lines)) > budget and len(lines) > 4:
            lines = lines[len(lines) // 2:]
        sections["memory"] = "\n".join(lines)

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
