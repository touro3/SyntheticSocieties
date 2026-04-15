"""Tests for decision/experimental_prompt_builder.py — prompt construction variants."""

from __future__ import annotations

from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.experimental_prompt_builder import (
    build_experimental_prompt,
    build_experimental_prompt_text,
    get_system_prompt,
)


def _profile() -> AgentProfile:
    return AgentProfile(
        agent_id="a0",
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="italy",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
        trust_people=0.6,
    )


def _state(**kw) -> AgentState:
    return AgentState(wealth=100.0, stress=kw.get("stress", 0.0), satisfaction=0.5)


def _memory() -> MemoryBuffer:
    return MemoryBuffer(max_items=5)


def _context() -> dict:
    return {
        "neighbors": ["a1"],
        "public_signal": {"economy": "stable"},
        "prices": {"food": 1.0},
        "resources": {"jobs": 100},
        "round_id": 1,
    }


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    def test_base_mode_returns_string(self):
        sp = get_system_prompt("base")
        assert isinstance(sp, str) and len(sp) > 0

    def test_balanced_mode_returns_string(self):
        sp = get_system_prompt("balanced")
        assert isinstance(sp, str) and len(sp) > 0

    def test_unknown_mode_returns_default(self):
        sp = get_system_prompt("nonexistent_mode")
        assert isinstance(sp, str) and len(sp) > 0


# ---------------------------------------------------------------------------
# build_experimental_prompt
# ---------------------------------------------------------------------------


class TestBuildExperimentalPrompt:
    def test_returns_list_of_dicts(self):
        msgs = build_experimental_prompt(_profile(), _state(), _memory(), _context(), round_id=1)
        assert isinstance(msgs, list)
        assert all(isinstance(m, dict) for m in msgs)

    def test_has_system_and_user_roles(self):
        msgs = build_experimental_prompt(_profile(), _state(), _memory(), _context(), round_id=1)
        roles = {m["role"] for m in msgs}
        assert "system" in roles
        assert "user" in roles

    def test_action_bounds_appear_in_user_content(self):
        msgs = build_experimental_prompt(_profile(), _state(), _memory(), _context(), round_id=1)
        user_text = next(m["content"] for m in msgs if m["role"] == "user")
        assert "work" in user_text.lower() or "action" in user_text.lower()

    def test_high_stress_adds_critical_note(self):
        high_stress_state = _state(stress=0.9)
        msgs = build_experimental_prompt(_profile(), high_stress_state, _memory(), _context(), round_id=1)
        user_text = " ".join(m["content"] for m in msgs)
        assert "stress" in user_text.lower()

    def test_no_stress_note_when_stress_is_low(self):
        low_stress_state = _state(stress=0.1)
        msgs = build_experimental_prompt(_profile(), low_stress_state, _memory(), _context(), round_id=1)
        user_text = " ".join(m["content"] for m in msgs)
        assert "critically high" not in user_text.lower()

    def test_population_context_included_when_provided(self):
        msgs = build_experimental_prompt(
            _profile(),
            _state(),
            _memory(),
            _context(),
            round_id=1,
            population_context="Trust in institutions is high.",
        )
        user_text = " ".join(m["content"] for m in msgs)
        assert "Trust in institutions" in user_text

    def test_population_context_excluded_when_flag_false(self):
        msgs = build_experimental_prompt(
            _profile(),
            _state(),
            _memory(),
            _context(),
            round_id=1,
            population_context="Trust in institutions is high.",
            use_population_context=False,
        )
        user_text = " ".join(m["content"] for m in msgs)
        assert "Trust in institutions" not in user_text

    def test_memory_excluded_when_flag_false(self):
        from agents.memory import MemoryItem

        mem = MemoryBuffer(max_items=5)
        mem.add(
            MemoryItem(
                round_id=0, partner_id=None, event_type="action", content="cooperated", outcome={}, importance=0.5
            )
        )
        msgs_with = build_experimental_prompt(
            _profile(), _state(), mem, _context(), round_id=1, use_memory_context=True
        )
        msgs_without = build_experimental_prompt(
            _profile(), _state(), mem, _context(), round_id=1, use_memory_context=False
        )
        text_with = " ".join(m["content"] for m in msgs_with)
        text_without = " ".join(m["content"] for m in msgs_without)
        assert len(text_without) <= len(text_with)

    def test_round_id_appears_in_output(self):
        msgs = build_experimental_prompt(_profile(), _state(), _memory(), _context(), round_id=7)
        user_text = " ".join(m["content"] for m in msgs)
        assert "7" in user_text

    def test_extra_guidance_appended(self):
        msgs = build_experimental_prompt(
            _profile(),
            _state(),
            _memory(),
            _context(),
            round_id=1,
            extra_guidance="Think carefully about long-term consequences.",
        )
        user_text = " ".join(m["content"] for m in msgs)
        assert "long-term" in user_text


# ---------------------------------------------------------------------------
# build_experimental_prompt_text
# ---------------------------------------------------------------------------


class TestBuildExperimentalPromptText:
    def test_returns_string(self):
        text = build_experimental_prompt_text(_profile(), _state(), _memory(), _context(), round_id=1)
        assert isinstance(text, str) and len(text) > 0
