"""Token budget management for LLM prompt construction.

A full BGF prompt (persona + ESS context + memory + social context + world
state) can approach or exceed the model's context window. Silent tokenizer
truncation cuts the persona block first — exactly the grounding we need most.

This module provides a lightweight, dependency-free character-based estimator
(4 chars ≈ 1 token for English text) and helpers to trim context sections in
priority order when the budget is tight.

Priority order (highest → lowest, i.e. last to drop):
  1. System prompt        — never trimmed
  2. State block          — never trimmed (current situation is critical)
  3. Population context   — RAG grounding (core experimental value)
  4. Social context       — GraphRAG (core experimental value)
  5. Persona block        — never trimmed
  6. Recent memory        — trim window if needed
  7. Extra guidance       — first to drop

Trim order (first to drop → last):
  1. extra guidance       — lowest value
  2. memory (halve)       — large, compressible
  3. social_context       — only if memory halving is insufficient
  4. population_context   — last RAG to drop

Per-model prompt budgets (prompt tokens only; excludes response headroom):
  - Mistral-7B-Instruct-v0.3  (32k ctx):  4096  — quality sweet spot for 7B attention
  - Qwen2.5-7B-Instruct       (131k ctx): 6144  — handles long context well
  - GPT-4o-mini               (128k ctx): 8192  — cost bounded by small agent count
  - default                               3072  — safe floor for unknown models
"""

from __future__ import annotations

import warnings

# Improved estimate: ~3.3 characters per token for English prose with
# numbers (validated against Mistral tokenizer). The old value of 4 was
# ~20% too generous, risking silent truncation at the model level.
_CHARS_PER_TOKEN = 3.3

# Default budget for unknown/unconfigured models.
DEFAULT_MAX_TOKENS = 3072

# Per-model prompt budgets keyed by model_id substring.
_MODEL_BUDGETS: dict[str, int] = {
    "mistral-7b": 4096,
    "mistral-7b-instruct": 4096,
    "qwen2.5-7b": 6144,
    "gpt-4o-mini": 8192,
}

# Optional: actual tokenizer for exact counting (set via set_tokenizer).
_tokenizer = None


def set_tokenizer(tokenizer) -> None:
    """Register a tokenizer for exact token counting.

    Call this after loading the model to switch from character-based
    estimation to exact tokenizer-based counting.
    """
    global _tokenizer
    _tokenizer = tokenizer


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
    """Estimate token count — uses tokenizer if available, else char heuristic."""
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass  # Fall through to heuristic
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


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
    Sections are dropped in priority order (first to drop = lowest value):
      1. extra guidance       — lowest value, drop first
      2. memory (halve)       — large and compressible
      3. social_context       — RAG, drop only if halving memory is insufficient
      4. population_context   — most valuable RAG, last to drop
    System, state, and persona are never touched.
    """
    fixed_tokens = (
        estimate_tokens(system)
        + estimate_tokens(persona)
        + estimate_tokens(state)
        + estimate_tokens(context)
        + 50  # formatting overhead
    )

    sections = {
        "memory": memory,
        "population_context": population_context,
        "social_context": social_context,
        "extra": extra,
    }

    def _variable_tokens() -> int:
        return sum(estimate_tokens(v or "") for v in sections.values())

    budget = max_tokens - fixed_tokens

    # Step 1: Drop extra guidance first (lowest value).
    if _variable_tokens() > budget and sections.get("extra"):
        sections["extra"] = None

    # Step 2: Halve memory repeatedly (large, compressible).
    if sections["memory"] and _variable_tokens() > budget:
        lines = sections["memory"].splitlines()
        while lines and _variable_tokens() > budget and len(lines) > 4:
            lines = lines[len(lines) // 2:]
            sections["memory"] = "\n".join(lines)

    # Step 3: Drop RAG contexts only as a last resort.
    rag_drop_order = ["social_context", "population_context"]
    for key in rag_drop_order:
        if _variable_tokens() <= budget:
            break
        if sections.get(key):
            sections[key] = None
            warnings.warn(
                f"token_budget: dropped '{key}' to fit within {max_tokens}-token limit. "
                "RAG context will be absent from this prompt.",
                stacklevel=3,
            )

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
