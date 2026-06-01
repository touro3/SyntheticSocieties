"""Tests for the opt-in disk-persistent semantic memory (ruflo AgentDB pattern)."""

from __future__ import annotations

from agents.memory import HierarchicalMemory, MemoryItem
from agents.persistent_memory import PersistentMemoryStore


def _item(round_id: int, content: str, event_type: str = "work") -> MemoryItem:
    return MemoryItem(
        round_id=round_id,
        partner_id=None,
        event_type=event_type,
        content=content,
        outcome={"wealth_delta": 8},
    )


def test_store_add_and_recall_ranks_by_similarity(tmp_path):
    db = tmp_path / "memory.db"
    store = PersistentMemoryStore(db)
    store.add(_item(1, "I cooperated with my neighbor and we both gained", "cooperate"))
    store.add(_item(2, "I worked hard alone for wealth"))
    store.add(_item(3, "I saved money for the future", "save"))
    store.flush()

    assert store.count() == 3
    results = store.recall("cooperation with neighbor", k=2)
    assert len(results) == 2
    # Most similar should be the cooperation memory.
    assert results[0]["event_type"] == "cooperate"
    assert "score" in results[0]
    store.close()


def test_store_persists_across_connections(tmp_path):
    db = tmp_path / "memory.db"
    s1 = PersistentMemoryStore(db)
    s1.add(_item(1, "persisted memory line"))
    s1.close()

    s2 = PersistentMemoryStore(db)
    assert s2.count() == 1
    assert s2.recall("persisted", k=1)[0]["content"] == "persisted memory line"
    s2.close()


def test_recall_empty_store_returns_empty():
    store = PersistentMemoryStore(":memory:")
    assert store.recall("anything") == []
    store.close()


def test_hierarchical_memory_default_has_no_persistent_store():
    """Research-validity guard: default construction must stay inert."""
    mem = HierarchicalMemory(max_recent=5)
    assert mem.persistent_store is None


def test_hierarchical_memory_mirrors_when_path_supplied(tmp_path):
    db = tmp_path / "agent.db"
    mem = HierarchicalMemory(max_recent=5, persistent_db_path=str(db))
    assert mem.persistent_store is not None

    mem.add(_item(1, "a cooperative round", "cooperate"))
    mem.add_batch([_item(2, "another working round")])
    mem.flush_pending()

    assert mem.persistent_store.count() == 2
    hits = mem.persistent_store.recall("cooperative", k=1)
    assert hits and hits[0]["event_type"] == "cooperate"


def test_in_memory_behavior_unchanged_when_inert():
    """Same sequence with/without an (absent) persistent store yields the
    same recent/archive contents — protects the M0–M3 ablation."""
    mem = HierarchicalMemory(max_recent=3)
    for i in range(6):
        mem.add(_item(i, f"round {i}"))
    assert len(mem.recent) == 3
    assert mem.recent[-1].round_id == 5
    assert mem.persistent_store is None
