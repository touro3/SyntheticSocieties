"""Hierarchical agent memory with automatic reflection summarization.

Three-tier architecture:
  recent   — sliding window of the last N events (fast access for prompts)
  archive  — older events evicted from recent (full episodic history)
  reflections — compressed natural-language summaries generated from archive

The reflection tier is what makes memory truly "hierarchical": once archive
grows beyond a threshold, old events are distilled into a single insight
string. This lets the LLM see a full career summary without exhausting its
context window on raw event lists.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    round_id: int
    partner_id: Optional[str]
    event_type: str
    content: str
    outcome: Dict[str, Any]


class HierarchicalMemory:
    # Compress archive into a reflection whenever it reaches this size.
    _COMPRESS_THRESHOLD = 20

    def __init__(self, max_recent: int = 20, archive_size: int = 100):
        self.max_recent = max_recent
        self.archive_size = archive_size
        self.recent: List[MemoryItem] = []
        self.archive: List[MemoryItem] = []
        self.reflections: List[str] = []

        # Cache: None means "not yet computed"; "" is a valid cached value.
        self._reflection_cache: Optional[str] = None
        self._cache_dirty: bool = True

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, item: MemoryItem) -> None:
        self.recent.append(item)
        self._cache_dirty = True

        if len(self.recent) > self.max_recent:
            evicted = self.recent.pop(0)
            self.archive.append(evicted)

            if len(self.archive) > self.archive_size:
                self.archive.pop(0)

            # Compress archive into a reflection periodically.
            if len(self.archive) % self._COMPRESS_THRESHOLD == 0:
                self._compress_archive()

    def _compress_archive(self) -> None:
        """Summarize the current archive into a reflection string and store it."""
        reflection = self._build_reflection_text(self.archive)
        if reflection:
            # Replace the last reflection if it exists (keeps list bounded to 1 entry).
            if self.reflections:
                self.reflections[-1] = reflection
            else:
                self.reflections.append(reflection)

    # ── Reflection generation ─────────────────────────────────────────────────

    def generate_reflection(self) -> str:
        """Return a natural-language reflection over all stored events.

        Rule-based — no LLM call needed. Analyzes action frequency, partner
        relationships, and recency to produce a concise history summary.
        Cached between calls; invalidated whenever a new event is added.
        """
        if not self._cache_dirty and self._reflection_cache is not None:
            return self._reflection_cache

        all_items = self.archive + self.recent
        self._reflection_cache = self._build_reflection_text(all_items)
        self._cache_dirty = False
        return self._reflection_cache

    @staticmethod
    def _build_reflection_text(items: List[MemoryItem]) -> str:
        """Convert a list of memory items into a concise summary string."""
        if not items:
            return ""

        action_counts: Counter[str] = Counter(m.event_type for m in items)
        total = len(items)

        # Action frequency summary
        top_actions = action_counts.most_common(3)
        action_summary = ", ".join(
            f"{action} ({count}×)" for action, count in top_actions
        )

        # Partner summary with reciprocation rates from outcome data
        partners: Counter[str] = Counter(
            m.partner_id for m in items
            if m.event_type == "cooperate" and m.partner_id
        )
        partner_summary = ""
        if partners:
            partner_details = []
            for partner, count in partners.most_common(3):
                # Check outcome data for reciprocation info
                partner_items = [
                    m for m in items
                    if m.event_type == "cooperate" and m.partner_id == partner
                ]
                reciprocated = sum(
                    1 for m in partner_items
                    if m.outcome.get("reciprocated") is True
                )
                total_coop = len(partner_items)
                if reciprocated > 0 or any(
                    "reciprocated" in m.outcome for m in partner_items
                ):
                    pct = round(100 * reciprocated / total_coop)
                    partner_details.append(
                        f"{partner} ({reciprocated}/{total_coop} reciprocated, {pct}%)"
                    )
                else:
                    partner_details.append(partner)
            partner_summary = (
                f" Most frequent cooperation partners: {', '.join(partner_details)}."
            )

        # Stress/outcome trend from most recent items (last 5)
        recent_slice = items[-5:]
        recent_actions = [m.event_type for m in recent_slice]
        recent_summary = ""
        if recent_actions:
            recent_summary = f" Recently: {', '.join(recent_actions)}."

        return (
            f"Over your recorded history ({total} events), you most often chose: "
            f"{action_summary}.{partner_summary}{recent_summary}"
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve(self, query: str | None = None, partner_id: str | None = None, limit: int = 5) -> List[MemoryItem]:
        """Search memory for relevant items by partner_id or keywords."""
        candidates = self.recent + self.archive

        if partner_id:
            results = [m for m in candidates if m.partner_id == partner_id]
            return results[-limit:]

        if query:
            keywords = query.lower().split()
            scored = []
            for m in candidates:
                score = sum(1 for kw in keywords if kw in m.content.lower())
                if score > 0:
                    scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored[:limit]]

        return self.recent[-limit:]

    def get_recent(self, limit: int = 5) -> List[MemoryItem]:
        """Return the most recent N items (used by prompt builder)."""
        return self.recent[-limit:]

    def get_full_context(self, limit: int = 10) -> List[MemoryItem]:
        """Return a blend of archive and recent events."""
        return (self.archive + self.recent)[-limit:]


class MemoryBuffer(HierarchicalMemory):
    """Compatibility alias for legacy tests."""

    def __init__(self, max_items: int = 50):
        super().__init__(max_recent=max_items)
