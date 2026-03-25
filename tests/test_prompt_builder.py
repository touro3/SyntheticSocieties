"""Tests for prompt builder — no GPU required."""
from agents.profile import AgentProfile
from agents.state import AgentState
from agents.memory import MemoryBuffer, MemoryItem
from decision.prompt_builder import (
    build_prompt,
    build_prompt_text,
    build_persona_block,
    build_state_block,
    build_memory_block,
    build_context_block,
    _level_word,
)


def _make_profile(**kwargs):
    defaults = dict(
        agent_id="agent_0", age=35, income=1000.0,
        education="college", occupation="worker", location="urban",
        political_preference="center", social_class="middle",
        trust_people=0.7, political_orientation=0.3,
        life_satisfaction=0.8, risk_tolerance=0.6,
        competitiveness=0.4, social_activity=0.5,
        gender=2, country="AT",
    )
    defaults.update(kwargs)
    return AgentProfile(**defaults)


def _make_state(**kwargs):
    defaults = dict(wealth=100.0, stress=0.5, satisfaction=0.3)
    defaults.update(kwargs)
    return AgentState(**defaults)


def _make_memory():
    mem = MemoryBuffer(max_items=5)
    mem.add(MemoryItem(round_id=0, event_type="work", partner_id=None, content="earned income", outcome={}))
    mem.add(MemoryItem(round_id=1, event_type="cooperate", partner_id="agent_1", content="shared resources", outcome={}))
    return mem


def _make_context():
    return {
        "world": {
            "prices": {"food": 1.0},
            "public_signal": {"economy": "stable"},
            "resources": {"jobs": 100.0},
        },
        "network": {"neighbors": ["agent_1", "agent_2"]},
    }


def test_persona_block_includes_key_attributes():
    profile = _make_profile()
    text = build_persona_block(profile)
    assert "agent_0" in text
    assert "35" in text
    assert "female" in text
    assert "AT" in text
    assert "trust" in text.lower()
    assert "left" in text.lower()  # political_orientation=0.3 → left-leaning


def test_state_block_format():
    state = _make_state()
    text = build_state_block(state)
    assert "wealth=100.0" in text
    assert "stress=0.50" in text


def test_memory_block_empty():
    mem = MemoryBuffer(max_items=5)
    text = build_memory_block(mem)
    assert "no memories" in text.lower()


def test_memory_block_with_items():
    mem = _make_memory()
    text = build_memory_block(mem)
    assert "work" in text
    assert "cooperate" in text
    assert "agent_1" in text


def test_context_block_includes_neighbors():
    context = _make_context()
    text = build_context_block(context)
    assert "agent_1" in text
    assert "agent_2" in text
    assert "Food price" in text


def test_build_prompt_returns_messages():
    profile = _make_profile()
    state = _make_state()
    memory = _make_memory()
    context = _make_context()

    messages = build_prompt(profile, state, memory, context, round_id=3)
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Round 3" in messages[1]["content"]
    assert "agent_0" in messages[1]["content"]


def test_build_prompt_text_string():
    profile = _make_profile()
    state = _make_state()
    memory = _make_memory()
    context = _make_context()

    text = build_prompt_text(profile, state, memory, context, round_id=5)
    assert isinstance(text, str)
    assert "[SYSTEM]" in text
    assert "[USER]" in text


def test_level_word():
    assert _level_word(0.1) == "very low"
    assert _level_word(0.3) == "low"
    assert _level_word(0.5) == "moderate"
    assert _level_word(0.7) == "high"
    assert _level_word(0.9) == "very high"
