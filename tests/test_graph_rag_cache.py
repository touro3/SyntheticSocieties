"""Tests for GraphRAG centrality caching correctness and performance."""

import pytest
from unittest.mock import patch
from decision.graph_rag import GraphRAG


def _cooperate_event(src: str, tgt: str, round_id: int = 1) -> dict:
    return {
        "agent_id": src,
        "action": {"action_type": "cooperate", "target_agent_id": tgt},
        "round_id": round_id,
    }


class TestCentralityCache:
    def test_cache_is_none_before_first_call(self):
        rag = GraphRAG()
        rag.add_event(_cooperate_event("a1", "a2"))
        assert rag._centrality_cache is None
        assert rag._cache_dirty is True

    def test_cache_populated_after_get_social_context(self):
        rag = GraphRAG()
        rag.add_event(_cooperate_event("a1", "a2"))
        rag.get_social_context("a1")
        assert rag._centrality_cache is not None
        assert rag._cache_dirty is False

    def test_cache_invalidated_on_add_event(self):
        rag = GraphRAG()
        rag.add_event(_cooperate_event("a1", "a2"))
        rag.get_social_context("a1")  # populate cache
        assert rag._cache_dirty is False

        rag.add_event(_cooperate_event("a2", "a3"))  # should dirty cache
        assert rag._cache_dirty is True

    def test_cache_invalidated_on_build_from_events(self, tmp_path):
        rag = GraphRAG()
        rag.add_event(_cooperate_event("a1", "a2"))
        rag.get_social_context("a1")
        assert rag._cache_dirty is False

        events_file = tmp_path / "events.jsonl"
        events_file.write_text("")
        rag.build_from_events(events_file)
        assert rag._cache_dirty is True

    def test_betweenness_computed_only_once_for_two_calls(self):
        rag = GraphRAG()
        for i in range(5):
            rag.add_event(_cooperate_event("a0", f"a{i+1}"))

        call_count = 0
        original = rag._compute_centrality

        def counting_compute():
            nonlocal call_count
            call_count += 1
            return original()

        rag._compute_centrality = counting_compute

        rag.get_social_context("a0")
        rag.get_social_context("a0")

        assert call_count == 1  # second call uses cache

    def test_cache_recomputed_after_dirty(self):
        rag = GraphRAG()
        for i in range(3):
            rag.add_event(_cooperate_event("a0", f"a{i+1}"))

        rag.get_social_context("a0")
        old_cache = rag._centrality_cache

        rag.add_event(_cooperate_event("a0", "a9"))  # dirties cache
        rag.get_social_context("a0")

        assert rag._centrality_cache is not old_cache  # new object

    def test_non_cooperative_event_does_not_dirty_cache(self):
        rag = GraphRAG()
        rag.add_event(_cooperate_event("a1", "a2"))
        rag.get_social_context("a1")
        assert rag._cache_dirty is False

        # Non-cooperation event
        rag.add_event({
            "agent_id": "a1",
            "action": {"action_type": "work"},
            "round_id": 2,
        })
        # Graph structure didn't change — cache should stay clean
        assert rag._cache_dirty is False
