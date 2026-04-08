"""Hierarchical agent memory with automatic reflection summarization.

Three-tier architecture:
  recent   — sliding window of the last N events (fast access for prompts)
  archive  — older events evicted from recent (full episodic history)
  reflections — compressed natural-language summaries generated from archive

The reflection tier is what makes memory truly "hierarchical": once archive
grows beyond a threshold, old events are distilled into a single insight
string. This lets the LLM see a full career summary without exhausting its
context window on raw event lists.

Anti-drift features (Phase 28):
  - Recency-weighted reflections: exponential decay prevents early
    hallucinations from permanently poisoning the action distribution.
  - Importance scoring: cooperation and persona-aligned events get
    priority when selecting which memories to surface in prompts.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    round_id: int
    partner_id: Optional[str]
    event_type: str
    content: str
    outcome: Dict[str, Any]
    importance: float = field(default=0.0)


class HierarchicalMemory:
    # Compress archive into a reflection whenever it reaches this size.
    _COMPRESS_THRESHOLD = 20

    # Recency half-life: after this many events, weight drops to 50%.
    _RECENCY_HALF_LIFE = 10

    # Importance bonus for social actions (cooperation builds society).
    _SOCIAL_IMPORTANCE = 0.3

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
        # Assign importance if not already set.
        if item.importance == 0.0:
            item.importance = self._score_importance(item)
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

    @staticmethod
    def _score_importance(item: MemoryItem) -> float:
        """Score an event's importance for memory retention.

        Social interactions (cooperate) and events with meaningful outcomes
        are scored higher so they survive memory compression.
        """
        score = 0.5  # baseline

        # Social actions are more memorable.
        if item.event_type == "cooperate":
            score += HierarchicalMemory._SOCIAL_IMPORTANCE
        elif item.event_type == "save":
            score += 0.1

        # Events with large wealth changes are memorable.
        if item.outcome:
            delta = abs(item.outcome.get("wealth_delta", 0))
            if delta >= 10:
                score += 0.2

            # Reciprocation outcomes are especially salient.
            if item.outcome.get("reciprocated") is True:
                score += 0.2

        return min(1.0, score)

    def _compress_archive(self) -> None:
        """Summarize the current archive into a reflection string and store it.

        Keeps up to MAX_REFLECTIONS entries so the LLM gets a multi-scale
        view of the agent's behavioral history, rather than losing all
        historical context on each compression.
        """
        reflection = self._build_reflection_text(self.archive)
        if reflection:
            self.reflections.append(reflection)
            # Keep bounded — drop oldest reflections beyond the cap
            MAX_REFLECTIONS = 3
            if len(self.reflections) > MAX_REFLECTIONS:
                self.reflections = self.reflections[-MAX_REFLECTIONS:]

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
        """Convert a list of memory items into a concise summary string.

        Uses recency-weighted proportional language ('work 50%, cooperate 33%')
        with exponential decay so that recent behavior counts more than early
        rounds. This prevents early hallucinations from permanently skewing
        the action distribution visible to the LLM.

        Includes outcome tracking and a counterfactual cue.
        """
        if not items:
            return ""

        total = len(items)
        half_life = HierarchicalMemory._RECENCY_HALF_LIFE
        decay = math.log(2) / max(half_life, 1)

        # Recency-weighted action distribution.
        # Weight of event i (0-indexed from oldest) = exp(decay * i).
        # Most recent item has the highest weight.
        weighted_counts: Counter[str] = Counter()
        total_weight = 0.0
        for i, m in enumerate(items):
            w = math.exp(decay * i)
            weighted_counts[m.event_type] += w
            total_weight += w

        action_parts = []
        for action, w in weighted_counts.most_common():
            pct = round(100 * w / total_weight) if total_weight > 0 else 0
            action_parts.append(f"{action} {pct}%")
        action_summary = ", ".join(action_parts)

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
                        f"{partner} (reciprocated {pct}% of the time)"
                    )
                else:
                    partner_details.append(partner)
            partner_summary = (
                f" Cooperation partners: {', '.join(partner_details)}."
            )

        # Outcome trend: wealth and stress changes from recent items
        recent_slice = items[-5:]
        outcome_parts = []
        wealth_deltas = [m.outcome.get("wealth_delta", 0) for m in recent_slice if m.outcome]
        if wealth_deltas:
            avg_delta = sum(wealth_deltas) / len(wealth_deltas)
            direction = "gaining" if avg_delta > 0 else "losing" if avg_delta < 0 else "stable"
            outcome_parts.append(f"wealth {direction}")
        recent_actions = [m.event_type for m in recent_slice]
        if recent_actions:
            outcome_parts.append(f"recent actions: {', '.join(recent_actions)}")
        outcome_summary = f" Trend: {'; '.join(outcome_parts)}." if outcome_parts else ""

        # Counterfactual cue to prevent action lock-in
        counterfactual = " Note: your past choices do not constrain your current decision."

        return (
            f"Over {total} events, your recency-weighted action distribution was: "
            f"{action_summary}.{partner_summary}{outcome_summary}{counterfactual}"
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

    def get_important_recent(self, limit: int = 5) -> List[MemoryItem]:
        """Return the N most important recent items, sorted chronologically.

        Combines recency (position in list) with the item's importance score
        to decide which memories to surface. This ensures that high-importance
        events (cooperation, large wealth changes) survive even when the
        window is small, preventing social amnesia in long runs.
        """
        if len(self.recent) <= limit:
            return list(self.recent)

        half_life = self._RECENCY_HALF_LIFE
        decay = math.log(2) / max(half_life, 1)
        n = len(self.recent)

        scored = []
        for i, item in enumerate(self.recent):
            recency_weight = math.exp(decay * (i - n + 1))  # 1.0 for most recent
            combined = 0.6 * recency_weight + 0.4 * item.importance
            scored.append((combined, i, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:limit]
        # Return in chronological order.
        selected.sort(key=lambda x: x[1])
        return [item for _, _, item in selected]

    def get_full_context(self, limit: int = 10) -> List[MemoryItem]:
        """Return a blend of archive and recent events."""
        return (self.archive + self.recent)[-limit:]

    def get_action_distribution(self, weighted: bool = True) -> Dict[str, float]:
        """Return the action distribution across all memory items.

        If weighted=True, applies recency weighting. Returns a dict of
        action_type -> proportion (sums to 1.0).
        """
        all_items = self.archive + self.recent
        if not all_items:
            return {}

        if not weighted:
            counts: Counter[str] = Counter(m.event_type for m in all_items)
            total = sum(counts.values())
            return {a: c / total for a, c in counts.items()}

        half_life = self._RECENCY_HALF_LIFE
        decay_rate = math.log(2) / max(half_life, 1)
        weighted_counts: Counter[str] = Counter()
        total_weight = 0.0
        for i, m in enumerate(all_items):
            w = math.exp(decay_rate * i)
            weighted_counts[m.event_type] += w
            total_weight += w

        if total_weight == 0:
            return {}
        return {a: w / total_weight for a, w in weighted_counts.items()}


class MemoryBuffer(HierarchicalMemory):
    """Compatibility alias for legacy tests."""

    def __init__(self, max_items: int = 50):
        super().__init__(max_recent=max_items)
