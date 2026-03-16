"""
Prompt perturbation module for BGF robustness testing.

Applies systematic perturbations to LLM prompts to test
sensitivity to prompt phrasing. Three modes:

  - rephrase:  Paraphrase persona (same semantics, different words)
  - shuffle:   Randomize order of persona attribute lines
  - noise:     Inject random distractor sentences

Usage:
    from decision.prompt_perturbation import apply_perturbation
    perturbed_messages = apply_perturbation(messages, mode="shuffle", seed=42)
"""

from __future__ import annotations

import random
import re
from copy import deepcopy


# Rephrase mapping: original pattern → alternatives
_REPHRASE_MAP = {
    r"You have (very low|low|moderate|high|very high) trust in other people":
        "Your level of interpersonal trust is {level}",
    r"Your risk tolerance is (very low|low|moderate|high|very high)":
        "When it comes to risk, you are {level}ly tolerant",
    r"Your competitiveness is (very low|low|moderate|high|very high)":
        "Your competitive drive is {level}",
    r"Your life satisfaction is (very low|low|moderate|high|very high)":
        "You feel {level}ly satisfied with life",
    r"Your happiness is (very low|low|moderate|high|very high)":
        "Your overall happiness level is {level}",
    r"Your social activity level is (very low|low|moderate|high|very high)":
        "Socially, you are {level}ly active",
    r"Politically, you are (left-leaning|center-left|centrist|center-right|right-leaning)":
        "Your political stance is {level}",
    r"You are (religious|not religious)":
        "Regarding religion, you are {level}",
}

# Distractor sentences for noise mode
_DISTRACTORS = [
    "The weather today is pleasant.",
    "Markets are fluctuating due to global events.",
    "A local festival is being planned in the community.",
    "New technology advancements are being discussed.",
    "Traffic conditions are normal in the area.",
    "A neighbor recently started a garden.",
    "The local library has new books available.",
    "Energy prices have been stable this quarter.",
    "A new café opened downtown last week.",
    "The community center is hosting a fundraiser.",
]


def apply_perturbation(
    messages: list[dict],
    mode: str,
    seed: int = 42,
) -> list[dict]:
    """
    Apply perturbation to prompt messages.

    Args:
        messages: Chat-format messages [{role, content}, ...]
        mode: One of "rephrase", "shuffle", "noise"
        seed: Random seed for reproducibility

    Returns:
        Perturbed copy of messages (originals unchanged).
    """
    if mode not in ("rephrase", "shuffle", "noise"):
        raise ValueError(f"Invalid perturbation mode: {mode}. Use: rephrase, shuffle, noise")

    rng = random.Random(seed)
    perturbed = deepcopy(messages)

    # Find the user message (persona is in there)
    for msg in perturbed:
        if msg["role"] == "user":
            msg["content"] = _perturb_user_content(msg["content"], mode, rng)
            break

    return perturbed


def _perturb_user_content(content: str, mode: str, rng: random.Random) -> str:
    """Apply perturbation to the user message content."""
    if mode == "rephrase":
        return _rephrase(content, rng)
    elif mode == "shuffle":
        return _shuffle(content, rng)
    elif mode == "noise":
        return _add_noise(content, rng)
    return content


def _rephrase(content: str, rng: random.Random) -> str:
    """Rephrase persona descriptions while preserving semantics."""
    result = content
    for pattern, template in _REPHRASE_MAP.items():
        match = re.search(pattern, result)
        if match:
            level = match.group(1)
            replacement = template.format(level=level)
            result = result[:match.start()] + replacement + result[match.end():]
    return result


def _shuffle(content: str, rng: random.Random) -> str:
    """Shuffle the order of persona attribute lines."""
    # Split into sections by double newline
    sections = content.split("\n\n")

    # Find the persona section (usually the second one after "Round N.")
    if len(sections) < 3:
        return content

    # The persona block is typically index 1
    persona_section = sections[1]
    persona_lines = [l for l in persona_section.split(". ") if l.strip()]

    if len(persona_lines) > 2:
        # Keep the first line (identity) fixed, shuffle the rest
        first = persona_lines[0]
        rest = persona_lines[1:]
        rng.shuffle(rest)
        sections[1] = ". ".join([first] + rest)

    return "\n\n".join(sections)


def _add_noise(content: str, rng: random.Random) -> str:
    """Inject 2-3 random distractor sentences into the persona."""
    sections = content.split("\n\n")

    if len(sections) < 3:
        return content

    # Pick 2-3 distractors
    n_distractors = rng.randint(2, 3)
    distractors = rng.sample(_DISTRACTORS, n_distractors)

    # Insert after persona section
    persona_section = sections[1]
    persona_section += " " + " ".join(distractors)
    sections[1] = persona_section

    return "\n\n".join(sections)
