from __future__ import annotations

import json
import re
from typing import Any


def build_fidelity_messages(
    profile_def: dict[str, Any],
    profile_text: str,
    target_items: list[dict[str, Any]],
    dataset_rag_context: str,
    include_justification: bool = False,
) -> list[dict[str, str]]:
    output_keys = ", ".join([f'"{item["prompt_label"]}"' for item in target_items])

    system_lines = [
        "You are simulating a stochastic human respondent drawn from a population distribution.",
        "Your goal is NOT to give the most likely answer, but to sample a plausible answer consistent with the profile.",

        "Humans are noisy and inconsistent: introduce natural variation.",
        "Do NOT collapse to average or neutral values.",
        "Use the full scale when appropriate, including extreme values.",
        "Different runs for the same profile should produce slightly different answers.",
        "Preserve tendencies from the profile, but vary intensity.",

        "You are sampling from a distribution, not predicting a mean.",
        "Before answering, internally sample a small random variation that slightly shifts responses up or down.",

        "Return ONLY a valid JSON object.",
        f"The JSON must contain exactly these keys: {output_keys}.",
        "Every value must be a number between 0.0 and 1.0.",
        "Do not add extra keys, markdown, explanations, or text outside the JSON.",
        "If you output anything after the JSON, your answer is invalid.",
    ]

    if include_justification:
        system_lines.append(
            "After the JSON, add a new line that starts with JUSTIFICATION: followed by one short explanation based only on the profile."
        )

    survey_lines = []
    for item in target_items:
        label = item["prompt_label"]

        # Fix semântico específico
        if item["prompt_label"] == "left_right":
            desc = (
                f'{item["description"]}. '
                "Scale: 0.0 = strongly left-wing, 0.5 = center, 1.0 = strongly right-wing. "

                "Interpretation guide:\n"
                "- Higher values (towards 1.0): more anti-immigration, more traditional, less egalitarian\n"
        "- Lower values (towards 0.0): more pro-immigration, more egalitarian, more progressive\n"

        "IMPORTANT: Align your answer with the profile tendencies and dataset context."
    )
        else:
            desc = (
                f'{item["description"]}. '
                "IMPORTANT: Higher values mean MORE of the trait described. Lower values mean LESS."
            )

        survey_lines.append(
            f'- "{label}": {desc} '
            f'Use a value from {item["scale_min"]:.1f} to {item["scale_max"]:.1f}. '
            f'IMPORTANT: Higher values mean MORE of the trait described. Lower values mean LESS. '
            f'Avoid defaulting to midpoints; vary responses realistically.'
        )

    user_lines = [
        "##DATASET_CONTEXT",
        dataset_rag_context,
        "",
        "##PROFILE",
        profile_text,
        "",
        "##SURVEY_ITEMS",
        *survey_lines,
        "",
        "Respond now with ONLY the JSON.",
    ]

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_fidelity_output(
    raw_text: str,
    target_items: list[dict[str, Any]]
) -> tuple[dict[str, float], str | None]:
    json_obj = _extract_json_object(raw_text)
    parsed = json.loads(json_obj)

    result: dict[str, float] = {}
    for item in target_items:
        key = item["prompt_label"]
        if key not in parsed:
            raise ValueError(f"Missing key in LLM JSON output: {key}")

        value = float(parsed[key])
        value = max(item["scale_min"], min(item["scale_max"], value))
        result[item["name"]] = value

    justification = None
    if "JUSTIFICATION:" in raw_text:
        justification = raw_text.split("JUSTIFICATION:", 1)[1].strip()

    return result, justification


def prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def _extract_json_object(text: str) -> str:
    stack = []
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if start is None:
                start = i
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    return text[start:i + 1]

    raise ValueError("No valid JSON object found.")