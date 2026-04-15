from __future__ import annotations

import json
from typing import Any


def build_fidelity_messages(
    profile_def: dict[str, Any],
    profile_text: str,
    target_items: list[dict[str, Any]],
    dataset_rag_context: str,
    include_justification: bool = False,
) -> list[dict[str, str]]:

    output_keys = '"internal_monologue", ' + ", ".join([f'"{item["prompt_label"]}"' for item in target_items])

    system_lines = [
        "You are an actor deeply adopting a specific demographic persona based on survey data.",
        "You must answer the survey exactly as this specific person would, based strictly on their profile.",
        "Do NOT act like an AI. Do NOT default to neutral, average, or politically correct answers.",
        "If the profile implies strong conservatism, strong liberalism, high distrust, or total trust, reflect that fully and without hesitation.",
        "Your responses must be a genuine psychological reflection of the persona's age, income, and background.",
        "Do not hedge, apologize, or balance your views to sound 'fair' or 'progressive'.",
        "Do not use phrases like 'but I also understand' or 'while also being'. Pick a stance and own it fully.",
        "Return ONLY a valid JSON object.",
        f"The JSON must contain exactly these keys: {output_keys}.",
        "The 'internal_monologue' must be a 1-2 sentence thought process IN THE FIRST PERSON ('I feel that...') explaining your worldview before you answer the numerical scales.",
        "The subsequent keys must be numbers between 0.0 and 1.0.",
        "Do not add extra markdown, explanations, or text outside the JSON.",
    ]

    survey_lines = []
    for item in target_items:
        label = item["prompt_label"]

        if item["prompt_label"] == "left_right":
            desc = (
                f"{item['description']}. "
                "Scale: 0.0 = Left-wing (progressive, favors social democracy and state intervention). "
                "0.5 = Center. "
                "1.0 = Right-wing (conservative, favors free markets and traditional values). "
                "IMPORTANT: Map your persona's background directly to this political spectrum without bias."
            )
        # --- AQUI ESTÁ A ALTERAÇÃO 3 ---
        elif item["prompt_label"] == "immigration_same_ethnicity":
            desc = (
                f"{item['description']}. "
                "Scale: 0.0 = Prefers strict cultural/demographic boundaries. "
                "1.0 = Prefers completely open demographic integration. "
                "IMPORTANT: Evaluate this abstractly based on the persona's background, without trying to sound 'inclusive'."
            )
        # -------------------------------
        else:
            desc = (
                f"{item['description']}. "
                "Scale: 0.0 = Absolute minimum/None. 1.0 = Absolute maximum/Completely. "
                "IMPORTANT: Translate your persona's monologue into an exact numerical intensity."
            )

        survey_lines.append(f'- "{label}": {desc}')

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
        "Adopt the persona. Write your internal monologue, then provide the numerical scores. Respond with ONLY the JSON.",
    ]

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_fidelity_output(raw_text: str, target_items: list[dict[str, Any]]) -> tuple[dict[str, float], str | None]:
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
                    return text[start : i + 1]

    raise ValueError("No valid JSON object found.")
