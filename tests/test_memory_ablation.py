"""Tests for Memory System Ablation.

Covers MemoryLevel enum and build_memory_block() behaviour at each level.
Run with: pytest tests/test_memory_ablation.py -v
"""

from __future__ import annotations

from agents.memory import HierarchicalMemory, MemoryItem, MemoryLevel
from decision.prompt_builder import build_memory_block

# ── Helpers ───────────────────────────────────────────────────────────────────


def _item(
    round_id: int, event_type: str = "work", partner_id: str = "peer", content: str = "", wealth_delta: float = 8.0
) -> MemoryItem:
    return MemoryItem(
        round_id=round_id,
        partner_id=partner_id,
        event_type=event_type,
        content=content,
        outcome={"wealth_delta": wealth_delta},
    )


def _make_memory(n_recent: int = 3, n_archive: int = 5) -> HierarchicalMemory:
    """Create a HierarchicalMemory pre-populated with items."""
    mem = HierarchicalMemory(max_recent=20, archive_size=100)
    for i in range(n_archive):
        mem.archive.append(_item(i, event_type="work"))
    for i in range(n_recent):
        mem.recent.append(
            _item(
                n_archive + i, event_type="cooperate", partner_id=f"agent_{i}", content="cooperated", wealth_delta=4.0
            )
        )
    return mem


# ── MemoryLevel enum ──────────────────────────────────────────────────────────


class TestMemoryLevelEnum:
    def test_enum_values(self):
        assert MemoryLevel.M0 == 0
        assert MemoryLevel.M1 == 1
        assert MemoryLevel.M2 == 2
        assert MemoryLevel.M3 == 3

    def test_enum_ordering(self):
        assert MemoryLevel.M0 < MemoryLevel.M1 < MemoryLevel.M2 < MemoryLevel.M3

    def test_all_four_levels_exist(self):
        assert len(list(MemoryLevel)) == 4

    def test_is_intenum(self):
        from enum import IntEnum

        assert issubclass(MemoryLevel, IntEnum)


# ── HierarchicalMemory.level attribute ───────────────────────────────────────


class TestMemoryLevelAttribute:
    def test_default_level_is_m3(self):
        mem = HierarchicalMemory()
        assert mem.level == MemoryLevel.M3

    def test_explicit_m0(self):
        mem = HierarchicalMemory(level=MemoryLevel.M0)
        assert mem.level == MemoryLevel.M0

    def test_explicit_m1(self):
        mem = HierarchicalMemory(level=MemoryLevel.M1)
        assert mem.level == MemoryLevel.M1

    def test_explicit_m2(self):
        mem = HierarchicalMemory(level=MemoryLevel.M2)
        assert mem.level == MemoryLevel.M2

    def test_explicit_m3(self):
        mem = HierarchicalMemory(level=MemoryLevel.M3)
        assert mem.level == MemoryLevel.M3

    def test_backwards_compat_no_level_arg(self):
        """HierarchicalMemory(max_recent=20) still works without level arg."""
        mem = HierarchicalMemory(max_recent=20)
        assert mem.level == MemoryLevel.M3

    def test_integer_level_accepted(self):
        """Level can be passed as plain int for config-driven usage."""
        mem = HierarchicalMemory(level=0)
        assert mem.level == MemoryLevel.M0


# ── build_memory_block() — M0 ─────────────────────────────────────────────────


class TestBuildMemoryBlockM0:
    def test_returns_placeholder_string(self):
        mem = _make_memory()
        result = build_memory_block(mem, level=MemoryLevel.M0)
        assert result == "No memory context available."

    def test_ignores_recent_items(self):
        mem = _make_memory(n_recent=5, n_archive=0)
        result = build_memory_block(mem, level=MemoryLevel.M0)
        assert "cooperate" not in result
        assert "Round" not in result

    def test_ignores_archive_items(self):
        mem = _make_memory(n_recent=0, n_archive=10)
        result = build_memory_block(mem, level=MemoryLevel.M0)
        assert "work" not in result

    def test_no_reflection_generated(self):
        mem = _make_memory(n_recent=3, n_archive=25)  # enough to trigger reflection
        result = build_memory_block(mem, level=MemoryLevel.M0)
        assert "Memory summary" not in result
        assert "action distribution" not in result


# ── build_memory_block() — M1 ─────────────────────────────────────────────────


class TestBuildMemoryBlockM1:
    def test_includes_recent_events(self):
        mem = _make_memory(n_recent=3, n_archive=0)
        result = build_memory_block(mem, level=MemoryLevel.M1)
        assert "cooperate" in result

    def test_no_reflection_text(self):
        mem = _make_memory(n_recent=3, n_archive=25)
        result = build_memory_block(mem, level=MemoryLevel.M1)
        assert "Memory summary" not in result
        assert "action distribution" not in result

    def test_no_archive_mention(self):
        mem = _make_memory(n_recent=3, n_archive=10)
        result = build_memory_block(mem, level=MemoryLevel.M1)
        assert "archive" not in result.lower()
        assert "older" not in result.lower()

    def test_respects_window_size(self):
        mem = _make_memory(n_recent=10, n_archive=0)
        result = build_memory_block(mem, window=3, level=MemoryLevel.M1)
        assert result.count("Round") <= 3

    def test_empty_memory_returns_fallback(self):
        mem = HierarchicalMemory()
        result = build_memory_block(mem, level=MemoryLevel.M1)
        assert "no memories" in result.lower()

    def test_round_ids_present(self):
        mem = _make_memory(n_recent=2, n_archive=0)
        result = build_memory_block(mem, level=MemoryLevel.M1)
        assert "Round" in result


# ── build_memory_block() — M2 ─────────────────────────────────────────────────


class TestBuildMemoryBlockM2:
    def test_includes_recent_events(self):
        mem = _make_memory(n_recent=3, n_archive=5)
        result = build_memory_block(mem, level=MemoryLevel.M2)
        assert "cooperate" in result

    def test_shows_archive_count_when_nonzero(self):
        mem = _make_memory(n_recent=3, n_archive=5)
        result = build_memory_block(mem, level=MemoryLevel.M2)
        assert "5" in result

    def test_archive_label_present(self):
        mem = _make_memory(n_recent=2, n_archive=7)
        result = build_memory_block(mem, level=MemoryLevel.M2)
        assert "archive" in result.lower() or "older" in result.lower()

    def test_no_reflection_text(self):
        mem = _make_memory(n_recent=3, n_archive=25)
        result = build_memory_block(mem, level=MemoryLevel.M2)
        assert "Memory summary" not in result
        assert "action distribution" not in result

    def test_zero_archive_no_archive_line(self):
        mem = _make_memory(n_recent=3, n_archive=0)
        result = build_memory_block(mem, level=MemoryLevel.M2)
        assert "archive" not in result.lower()
        assert "older" not in result.lower()

    def test_empty_recent_with_archive(self):
        mem = _make_memory(n_recent=0, n_archive=5)
        result = build_memory_block(mem, level=MemoryLevel.M2)
        assert "5" in result  # archive count still shown


# ── build_memory_block() — M3 ─────────────────────────────────────────────────


class TestBuildMemoryBlockM3:
    def test_includes_reflection_when_items_exist(self):
        mem = _make_memory(n_recent=3, n_archive=5)
        result = build_memory_block(mem, level=MemoryLevel.M3)
        assert "Memory summary" in result

    def test_includes_recent_events(self):
        mem = _make_memory(n_recent=3, n_archive=0)
        result = build_memory_block(mem, level=MemoryLevel.M3)
        assert "cooperate" in result

    def test_m3_matches_no_level_arg(self):
        """Default (no level arg) must produce same output as explicit M3."""
        mem = _make_memory(n_recent=3, n_archive=5)
        result_explicit = build_memory_block(mem, level=MemoryLevel.M3)
        result_default = build_memory_block(mem)
        assert result_explicit == result_default


# ── Level read from memory.level attribute ────────────────────────────────────


class TestBuildMemoryBlockLevelFromMemory:
    def test_reads_m0_from_memory_attribute(self):
        mem = HierarchicalMemory(level=MemoryLevel.M0)
        mem.add(_item(1, "cooperate", content="test"))
        result = build_memory_block(mem)
        assert result == "No memory context available."

    def test_reads_m1_from_memory_attribute(self):
        mem = HierarchicalMemory(level=MemoryLevel.M1)
        mem.add(_item(1, "cooperate", content="test"))
        result = build_memory_block(mem)
        assert "cooperate" in result
        assert "Memory summary" not in result

    def test_explicit_level_arg_overrides_memory_level(self):
        """Explicit level= kwarg takes precedence over memory.level."""
        mem = HierarchicalMemory(level=MemoryLevel.M0)
        mem.add(_item(1, "cooperate", content="test"))
        # Override M0 with M1 via explicit arg
        result = build_memory_block(mem, level=MemoryLevel.M1)
        assert "cooperate" in result  # M1 shows recent events
        assert result != "No memory context available."
