"""Tests for memory-activation diagnostics (Phase 3).

Guarantees:
  - citation flag is correct: items rendered into the block count as cited;
    nothing is cited when the block suppresses memory (M0);
  - retrieval count tracks the memory level (M0=0, M1≤window, M3 importance);
  - the diagnostics wrapper is a pure pass-through (identical text) and the
    JSONL logger writes one record per call.
"""

from __future__ import annotations

import json

from agents.memory import HierarchicalMemory, MemoryItem, MemoryLevel
from bgf_logging.memory_diagnostics import (
    MemoryDiagnosticsLogger,
    build_memory_block_with_diagnostics,
    memory_activation_record,
)
from decision.prompt_builder import build_memory_block


def _mem(level: MemoryLevel, n: int = 6) -> HierarchicalMemory:
    m = HierarchicalMemory(max_recent=20, level=level)
    for i in range(n):
        m.add(
            MemoryItem(
                round_id=i,
                partner_id=f"agent_{i}",
                event_type="cooperate" if i % 2 else "work",
                content=f"event {i}",
                outcome={"reciprocated": bool(i % 2)},
                importance=0.5 + 0.05 * i,
            )
        )
    return m


def test_citation_rate_high_when_rendered():
    m = _mem(MemoryLevel.M1)
    text = build_memory_block(m, window=5, level=MemoryLevel.M1)
    rec = memory_activation_record(m, 5, MemoryLevel.M1, text)
    assert rec["retrieval_count"] > 0
    # M1 renders every retrieved item verbatim → full citation.
    assert rec["citation_rate"] == 1.0
    assert rec["citation_count"] == rec["retrieval_count"]


def test_no_citation_when_memory_suppressed():
    m = _mem(MemoryLevel.M0)
    text = build_memory_block(m, window=5, level=MemoryLevel.M0)
    rec = memory_activation_record(m, 5, MemoryLevel.M0, text)
    assert rec["retrieval_count"] == 0
    assert rec["citation_rate"] == 0.0


def test_retrieval_count_tracks_level():
    n0 = memory_activation_record(_mem(MemoryLevel.M0), 5, MemoryLevel.M0, "")["retrieval_count"]
    n1 = memory_activation_record(_mem(MemoryLevel.M1), 5, MemoryLevel.M1, "")["retrieval_count"]
    n3 = memory_activation_record(_mem(MemoryLevel.M3), 5, MemoryLevel.M3, "")["retrieval_count"]
    assert n0 == 0
    assert 0 < n1 <= 5
    assert 0 < n3 <= 5


def test_wrapper_is_pure_passthrough():
    m = _mem(MemoryLevel.M3)
    plain = build_memory_block(m, window=5, level=MemoryLevel.M3)
    wrapped = build_memory_block_with_diagnostics(m, window=5, level=MemoryLevel.M3)
    assert wrapped == plain


def test_logger_writes_one_record_per_call(tmp_path):
    sink = MemoryDiagnosticsLogger(tmp_path / "memdiag.jsonl")
    m = _mem(MemoryLevel.M1)
    for r in range(3):
        build_memory_block_with_diagnostics(
            m, window=5, level=MemoryLevel.M1, logger_sink=sink, round_id=r, agent_id="agent_x"
        )
    sink.close()
    lines = (tmp_path / "memdiag.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["agent_id"] == "agent_x"
    assert {"retrieval_count", "citation_rate", "memory_level", "retrieved_refs"} <= set(first)
