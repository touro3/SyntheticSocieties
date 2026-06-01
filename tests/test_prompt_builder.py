"""Tests for prompt builder — no GPU required."""

from agents.memory import MemoryBuffer, MemoryItem
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.prompt_builder import (
    _level_word,
    build_context_block,
    build_memory_block,
    build_persona_block,
    build_prompt,
    build_prompt_text,
    build_state_block,
)


def _make_profile(**kwargs):
    defaults = dict(
        agent_id="agent_0",
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="urban",
        political_preference="center",
        social_class="middle",
        trust_people=0.7,
        political_orientation=0.3,
        life_satisfaction=0.8,
        risk_tolerance=0.6,
        competitiveness=0.4,
        social_activity=0.5,
        gender=2,
        country="AT",
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
    mem.add(
        MemoryItem(round_id=1, event_type="cooperate", partner_id="agent_1", content="shared resources", outcome={})
    )
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


# ── max_tokens parameter ──────────────────────────────────────────────────────


class TestBuildPromptMaxTokens:
    """build_prompt and build_prompt_text accept and thread max_tokens correctly."""

    def _profile(self):
        return _make_profile()

    def _state(self):
        return _make_state()

    def _memory(self):
        return _make_memory()

    def _context(self):
        return _make_context()

    def test_max_tokens_accepted(self):
        msgs = build_prompt(
            self._profile(),
            self._state(),
            self._memory(),
            self._context(),
            round_id=1,
            max_tokens=4096,
        )
        assert isinstance(msgs, list)
        assert len(msgs) >= 2

    def test_max_tokens_none_uses_default(self):
        msgs_default = build_prompt(
            self._profile(),
            self._state(),
            self._memory(),
            self._context(),
            round_id=1,
        )
        msgs_none = build_prompt(
            self._profile(),
            self._state(),
            self._memory(),
            self._context(),
            round_id=1,
            max_tokens=None,
        )
        # Both paths should produce the same structure
        assert len(msgs_default) == len(msgs_none)

    def test_large_budget_preserves_rag_contexts(self):
        msgs = build_prompt(
            self._profile(),
            self._state(),
            self._memory(),
            self._context(),
            round_id=1,
            population_context="ESS population context: high trust in Austria.",
            social_context="Agent cooperated with neighbors recently.",
            max_tokens=4096,
        )
        full_text = " ".join(m["content"] for m in msgs)
        assert "ESS population context" in full_text
        assert "cooperated with neighbors" in full_text

    def test_build_prompt_text_max_tokens_threaded(self):
        text = build_prompt_text(
            self._profile(),
            self._state(),
            self._memory(),
            self._context(),
            round_id=1,
            population_context="Population: high trust cohort.",
            max_tokens=4096,
        )
        assert isinstance(text, str)
        assert "Population: high trust cohort." in text

    def test_all_optional_fields_none_still_valid(self):
        msgs = build_prompt(
            self._profile(),
            self._state(),
            self._memory(),
            self._context(),
            round_id=1,
            social_context=None,
            population_context=None,
            max_tokens=4096,
        )
        assert isinstance(msgs, list)
        assert all("role" in m and "content" in m for m in msgs)


# ── Position-bias mitigation (Permutation Invariance) ─────────────────────────

import re

from decision.output_parser import parse_llm_output


class TestPositionBiasMitigation:
    """System prompt action options are shuffled to prevent position bias."""

    def test_action_options_shuffled_across_builds(self):
        """Multiple calls with identical data should produce different action orderings."""
        from decision.system_prompts import get_shuffled_system_prompt

        orderings = set()
        for _ in range(30):
            prompt = get_shuffled_system_prompt()
            # Extract the action_type options pattern: <work|save|cooperate> (in any order)
            match = re.search(r'"action_type":\s*"<([^>]+)>"', prompt)
            assert match, "System prompt must contain action_type options"
            orderings.add(match.group(1))

        # With 3! = 6 permutations, 30 draws should produce at least 2 orderings
        assert len(orderings) >= 2, f"Expected shuffled orderings, but got only: {orderings}"

    def test_all_valid_actions_present(self):
        """Every shuffled prompt must contain all valid actions."""
        from decision.system_prompts import get_shuffled_system_prompt

        # Core actions in the prompt (communicate is not included in the
        # shuffled system prompt as it's a meta-action, not an economic action)
        core_actions = {"work", "save", "cooperate"}
        for _ in range(10):
            prompt = get_shuffled_system_prompt()
            for action in core_actions:
                assert action in prompt, f"Missing '{action}' in shuffled prompt"

    def test_build_prompt_uses_shuffled_actions(self):
        """build_prompt() should produce different action orderings across agents/rounds.

        The shuffle is seeded by (round_id, agent_id) so repeated calls with the
        same inputs are deterministic (prompt-log consistency), but different
        agents or different rounds produce different orderings.
        """
        import re as _re

        state = _make_state()
        memory = _make_memory()
        context = _make_context()

        orderings = set()
        # Vary both round_id and agent_id to exercise the shuffle seed space.
        for round_id in range(1, 16):
            for agent_suffix in ["_a", "_b"]:
                profile = _make_profile()
                profile.agent_id = f"agent{agent_suffix}_{round_id}"
                msgs = build_prompt(profile, state, memory, context, round_id=round_id)
                system = msgs[0]["content"]
                match = _re.search(r'"action_type":\s*"<([^>]+)>"', system)
                assert match, "System prompt in build_prompt must contain action options"
                orderings.add(match.group(1))

        # With 30 distinct (round_id, agent_id) pairs we expect multiple orderings.
        assert len(orderings) >= 2, f"Expected multiple action orderings across agents/rounds, got: {orderings}"

    def test_parser_handles_any_action_order(self):
        """output_parser reads action_type correctly regardless of prompt ordering."""
        for action in ["work", "save", "cooperate"]:
            if action == "cooperate":
                raw = f'{{"action_type": "{action}", "target_agent_id": "agent_1", "amount": 5, "reasoning_summary": "test"}}'
            else:
                raw = f'{{"action_type": "{action}", "amount": 5, "reasoning_summary": "test"}}'
            result, meta = parse_llm_output(raw, neighbors=["agent_1"])
            assert result is not None, f"Parser failed for action '{action}'"
            assert result.action_type == action

    def test_action_mechanics_block_shuffled(self):
        """The 'Action mechanics' list order should also be shuffled."""
        from decision.system_prompts import get_shuffled_system_prompt

        first_actions = []
        for _ in range(30):
            prompt = get_shuffled_system_prompt()
            # Find the first action after "Action mechanics:"
            match = re.search(r'Action mechanics:\n- "(\w+)"', prompt)
            assert match, "Must have Action mechanics block"
            first_actions.append(match.group(1))

        # Should see different first actions across 30 draws
        assert len(set(first_actions)) >= 2
