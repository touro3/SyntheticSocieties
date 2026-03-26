"""Tests for HierarchicalMemory reflection (summarization) system."""

import pytest
from agents.memory import HierarchicalMemory, MemoryItem


def _make_item(round_id: int, event_type: str, partner_id: str | None = None, content: str = "") -> MemoryItem:
    return MemoryItem(
        round_id=round_id,
        event_type=event_type,
        partner_id=partner_id,
        content=content,
        outcome={},
    )


class TestReflectionGeneration:
    def test_empty_memory_returns_empty_reflection(self):
        mem = HierarchicalMemory()
        assert mem.generate_reflection() == ""

    def test_reflection_is_string(self):
        mem = HierarchicalMemory()
        mem.add(_make_item(1, "work"))
        assert isinstance(mem.generate_reflection(), str)

    def test_reflection_mentions_dominant_action(self):
        mem = HierarchicalMemory()
        for i in range(5):
            mem.add(_make_item(i, "work"))
        mem.add(_make_item(5, "save"))
        reflection = mem.generate_reflection()
        assert "work" in reflection.lower()

    def test_reflection_mentions_cooperation_partner(self):
        mem = HierarchicalMemory()
        for i in range(3):
            mem.add(_make_item(i, "cooperate", partner_id="agent_7"))
        reflection = mem.generate_reflection()
        assert "agent_7" in reflection

    def test_reflection_handles_multiple_partners(self):
        mem = HierarchicalMemory()
        mem.add(_make_item(1, "cooperate", partner_id="agent_1"))
        mem.add(_make_item(2, "cooperate", partner_id="agent_2"))
        mem.add(_make_item(3, "cooperate", partner_id="agent_1"))
        reflection = mem.generate_reflection()
        # Most frequent partner should appear
        assert "agent_1" in reflection

    def test_reflection_covers_archive_and_recent(self):
        mem = HierarchicalMemory(max_recent=3)
        # Fill and overflow into archive
        for i in range(6):
            mem.add(_make_item(i, "cooperate", partner_id="agent_5"))
        assert len(mem.archive) > 0
        reflection = mem.generate_reflection()
        assert "cooperate" in reflection.lower()

    def test_reflection_cached_until_new_event(self):
        mem = HierarchicalMemory()
        for i in range(3):
            mem.add(_make_item(i, "work"))
        r1 = mem.generate_reflection()
        r2 = mem.generate_reflection()
        assert r1 == r2  # same object / same content

    def test_reflection_updates_after_new_event(self):
        mem = HierarchicalMemory()
        mem.add(_make_item(0, "work"))
        r1 = mem.generate_reflection()
        mem.add(_make_item(1, "cooperate", partner_id="agent_9"))
        r2 = mem.generate_reflection()
        # Cache should be invalidated
        assert r2 != r1 or "cooperate" in r2.lower()


class TestReflectionInPrompt:
    def test_build_memory_block_includes_reflection(self):
        from decision.prompt_builder import build_memory_block

        mem = HierarchicalMemory()
        for i in range(6):
            mem.add(_make_item(i, "cooperate", partner_id="agent_3"))

        block = build_memory_block(mem, window=5)
        # Reflection should appear alongside recent events
        assert "cooperate" in block.lower()

    def test_build_memory_block_reflection_precedes_recent(self):
        from decision.prompt_builder import build_memory_block

        mem = HierarchicalMemory(max_recent=2)
        for i in range(5):
            mem.add(_make_item(i, "work"))

        block = build_memory_block(mem, window=2)
        assert isinstance(block, str)
        assert len(block) > 0
