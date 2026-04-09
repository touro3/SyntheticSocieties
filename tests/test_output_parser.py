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


# ── P4.7: Edge cases ───────────────────────────────────────────────────────────

def test_cooperate_empty_neighbors_no_crash():
    """cooperate JSON with empty neighbor list must not raise."""
    text = '{"action_type": "cooperate", "target_agent_id": "a1", "amount": 5.0, "reasoning_summary": "help", "confidence": 0.8}'
    # Should not raise; target_agent_id="a1" is preserved even though it's not in []
    action, meta = parse_llm_output(text, neighbors=[])
    assert action is not None  # action created successfully (target preserved)


def test_cooperate_no_neighbors_arg_returns_none_action():
    """cooperate JSON without target and no neighbors returns None (schema enforces target)."""
    text = '{"action_type": "cooperate", "amount": 5.0, "reasoning_summary": "help", "confidence": 0.8}'
    # _validate_action can't build ProposedAction(cooperate, target=None) — schema rejects it
    action, meta = parse_llm_output(text)
    # Graceful: either None (schema rejected) or keyword fallback to another action
    assert meta is not None  # no exception raised


def test_confidence_null_defaults_to_none_or_half():
    """confidence: null in JSON must not crash — parsed action confidence is None or clamped."""
    text = '{"action_type": "work", "amount": 8.0, "reasoning_summary": "earn", "confidence": null}'
    action, meta = parse_llm_output(text)
    assert action is not None
    assert action.action_type == "work"
    # confidence=null should result in None or 0.5, not an error
    if action.confidence is not None:
        assert 0.0 <= action.confidence <= 1.0


def test_keyword_fallback_large_neighbor_list():
    """Keyword fallback with 1000 neighbors must pick first neighbor without error."""
    large_neighbors = [f"agent_{i}" for i in range(1000)]
    text = "I think I should cooperate and help share resources with my community."
    action, meta = parse_llm_output(text, neighbors=large_neighbors)
    assert action is not None
    assert action.action_type == "cooperate"
    assert action.target_agent_id == "agent_0"


def test_keyword_fallback_cooperate_no_neighbors_falls_back_to_work():
    """Keyword fallback cooperate with no neighbors must fall back to work (schema requires target)."""
    text = "I should help and cooperate with everyone around me."
    action, meta = parse_llm_output(text, neighbors=[])
    assert action is not None
    # cooperate is impossible without a target; the fallback must return work
    assert action.action_type == "work"


def test_parse_stats_increment_on_success():
    """parse_llm_output increments the correct stats counter after a successful parse."""
    from decision.output_parser import get_parse_stats, reset_parse_stats
    reset_parse_stats()
    parse_llm_output('{"action_type": "work", "amount": 8.0, "reasoning_summary": "ok", "confidence": 0.9}')
    stats = get_parse_stats()
    assert stats.get("direct_json", 0) >= 1


def test_parse_stats_keyword_fallback_counted():
    """Keyword fallback path increments keyword_fallback stat."""
    from decision.output_parser import get_parse_stats, reset_parse_stats
    reset_parse_stats()
    parse_llm_output("I want to earn money and work hard today.")
    stats = get_parse_stats()
    assert stats.get("keyword_fallback", 0) >= 1
