"""Tests for output parser — no GPU required."""
from decision.output_parser import parse_llm_output


def test_parse_valid_json():
    text = '{"action_type": "work", "amount": 10.0, "reasoning_summary": "need income", "confidence": 0.9}'
    action, meta = parse_llm_output(text)
    assert action is not None
    assert action.action_type == "work"
    assert action.amount == 10.0
    assert meta["parse_method"] == "direct_json"
    assert meta["parse_success"] is True


def test_parse_json_with_extra_text():
    text = 'Sure! Here is my choice:\n{"action_type": "save", "amount": 5.0, "reasoning_summary": "saving for future", "confidence": 0.7}\nThank you!'
    action, meta = parse_llm_output(text)
    assert action is not None
    assert action.action_type == "save"
    assert meta["parse_method"] == "regex_json"


def test_parse_cooperate_with_target():
    text = '{"action_type": "cooperate", "target_agent_id": "agent_1", "amount": 5.0, "reasoning_summary": "help neighbor", "confidence": 0.8}'
    action, meta = parse_llm_output(text, neighbors=["agent_1", "agent_2"])
    assert action is not None
    assert action.action_type == "cooperate"
    assert action.target_agent_id == "agent_1"


def test_parse_cooperate_invalid_target_corrected():
    text = '{"action_type": "cooperate", "target_agent_id": "agent_99", "amount": 5.0, "reasoning_summary": "help", "confidence": 0.8}'
    action, meta = parse_llm_output(text, neighbors=["agent_1", "agent_2"])
    assert action is not None
    assert action.target_agent_id == "agent_1"  # Corrected to first valid neighbor


def test_parse_invalid_action_type():
    text = '{"action_type": "attack", "amount": 10.0, "reasoning_summary": "war", "confidence": 0.9}'
    action, meta = parse_llm_output(text)
    assert action is None
    assert "Invalid action_type" in meta.get("parse_error", "")


def test_parse_keyword_fallback_cooperate():
    text = "I think I should help my neighbor and share resources with them."
    action, meta = parse_llm_output(text, neighbors=["agent_3"])
    assert action is not None
    assert action.action_type == "cooperate"
    assert meta["parse_method"] == "keyword_fallback"


def test_parse_keyword_fallback_work():
    text = "I need to earn more money to survive."
    action, meta = parse_llm_output(text)
    assert action is not None
    assert action.action_type == "work"


def test_parse_keyword_fallback_save():
    text = "I should conserve my resources and save for the future."
    action, meta = parse_llm_output(text)
    assert action is not None
    assert action.action_type == "save"


def test_parse_empty_output():
    action, meta = parse_llm_output("")
    assert action is None
    assert "Empty" in meta["parse_error"]


def test_parse_none_output():
    action, meta = parse_llm_output(None)
    assert action is None


def test_amount_clamping():
    text = '{"action_type": "work", "amount": 999.0, "reasoning_summary": "greedy", "confidence": 0.9}'
    action, _ = parse_llm_output(text)
    assert action.amount == 20.0  # Clamped to max


def test_confidence_clamping():
    text = '{"action_type": "save", "amount": 5.0, "reasoning_summary": "safe", "confidence": 5.0}'
    action, _ = parse_llm_output(text)
    assert action.confidence == 1.0  # Clamped to max
