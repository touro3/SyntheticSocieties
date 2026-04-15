"""Tests for decision/fidelity_prompt_builder.py — survey prompt construction and parsing."""

from __future__ import annotations

import json

import pytest

from decision.fidelity_prompt_builder import (
    _extract_json_object,
    build_fidelity_messages,
    parse_fidelity_output,
    prompt_text,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILE_DEF = {
    "agent_id": "a0",
    "age": 35,
    "income": 1000.0,
    "country": "DE",
}

PROFILE_TEXT = "Age 35, middle-class German worker."

TARGET_ITEMS = [
    {
        "name": "trust_people",
        "prompt_label": "trust_people",
        "description": "How much do you trust most people?",
        "scale_min": 0.0,
        "scale_max": 1.0,
    },
    {
        "name": "life_satisfaction",
        "prompt_label": "life_satisfaction",
        "description": "How satisfied are you with your life?",
        "scale_min": 0.0,
        "scale_max": 1.0,
    },
]

DATASET_CONTEXT = "ESS-11: Average trust in Germany is 0.6."


# ---------------------------------------------------------------------------
# build_fidelity_messages
# ---------------------------------------------------------------------------


class TestBuildFidelityMessages:
    def test_returns_two_messages(self):
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, TARGET_ITEMS, DATASET_CONTEXT)
        assert len(msgs) == 2

    def test_first_is_system(self):
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, TARGET_ITEMS, DATASET_CONTEXT)
        assert msgs[0]["role"] == "system"

    def test_second_is_user(self):
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, TARGET_ITEMS, DATASET_CONTEXT)
        assert msgs[1]["role"] == "user"

    def test_all_target_labels_in_system_prompt(self):
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, TARGET_ITEMS, DATASET_CONTEXT)
        system_text = msgs[0]["content"]
        for item in TARGET_ITEMS:
            assert item["prompt_label"] in system_text

    def test_dataset_context_in_user_prompt(self):
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, TARGET_ITEMS, DATASET_CONTEXT)
        user_text = msgs[1]["content"]
        assert "ESS-11" in user_text

    def test_profile_text_in_user_prompt(self):
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, TARGET_ITEMS, DATASET_CONTEXT)
        user_text = msgs[1]["content"]
        assert PROFILE_TEXT in user_text

    def test_left_right_scale_description(self):
        left_right_item = {
            "name": "left_right",
            "prompt_label": "left_right",
            "description": "Political orientation.",
            "scale_min": 0.0,
            "scale_max": 1.0,
        }
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, [left_right_item], DATASET_CONTEXT)
        user_text = msgs[1]["content"]
        assert "Left-wing" in user_text or "left" in user_text.lower()

    def test_immigration_item_has_custom_scale(self):
        item = {
            "name": "immigration_same_ethnicity",
            "prompt_label": "immigration_same_ethnicity",
            "description": "Immigration attitudes.",
            "scale_min": 0.0,
            "scale_max": 1.0,
        }
        msgs = build_fidelity_messages(PROFILE_DEF, PROFILE_TEXT, [item], DATASET_CONTEXT)
        user_text = msgs[1]["content"]
        assert "demographic" in user_text.lower() or "integration" in user_text.lower()


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_extracts_simple_object(self):
        text = 'Here is the answer: {"key": "value"} done.'
        extracted = _extract_json_object(text)
        assert json.loads(extracted) == {"key": "value"}

    def test_extracts_nested_object(self):
        text = '{"outer": {"inner": 42}}'
        extracted = _extract_json_object(text)
        assert json.loads(extracted)["outer"]["inner"] == 42

    def test_raises_when_no_json(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            _extract_json_object("no json here at all")


# ---------------------------------------------------------------------------
# parse_fidelity_output
# ---------------------------------------------------------------------------


class TestParseFidelityOutput:
    def test_parses_valid_json(self):
        raw = json.dumps({"trust_people": 0.7, "life_satisfaction": 0.5, "internal_monologue": "I feel ok."})
        result, justification = parse_fidelity_output(raw, TARGET_ITEMS)
        assert result["trust_people"] == pytest.approx(0.7)
        assert result["life_satisfaction"] == pytest.approx(0.5)

    def test_clamps_values_to_scale(self):
        raw = json.dumps({"trust_people": 1.5, "life_satisfaction": -0.3})
        result, _ = parse_fidelity_output(raw, TARGET_ITEMS)
        assert result["trust_people"] == pytest.approx(1.0)
        assert result["life_satisfaction"] == pytest.approx(0.0)

    def test_raises_on_missing_key(self):
        raw = json.dumps({"trust_people": 0.7})  # missing life_satisfaction
        with pytest.raises((ValueError, KeyError)):
            parse_fidelity_output(raw, TARGET_ITEMS)


# ---------------------------------------------------------------------------
# prompt_text
# ---------------------------------------------------------------------------


class TestPromptText:
    def test_renders_system_and_user(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Answer this."},
        ]
        text = prompt_text(msgs)
        assert "[SYSTEM]" in text
        assert "[USER]" in text
        assert "You are helpful." in text
        assert "Answer this." in text
