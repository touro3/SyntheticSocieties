"""Hierarchical agent memory with automatic reflection summarization.

Three-tier architecture:
  recent   — sliding window of the last N events (fast access for prompts)
  archive  — older events evicted from recent (full episodic history)
  reflections — compressed natural-language summaries generated from archive

The reflection tier is what makes memory truly "hierarchical": once archive
grows beyond a threshold, old events are distilled into a single insight
string. This lets the LLM see a full career summary without exhausting its
context window on raw event lists.

Anti-drift features:
  - Recency-weighted reflections: exponential decay prevents early
    hallucinations from permanently poisoning the action distribution.
  - Importance scoring: cooperation and persona-aligned events get
    priority when selecting which memories to surface in prompts.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryLevel(IntEnum):
    """Memory depth surfaced to the LLM — used for the memory ablation study.

    M0 — no memory context (LLM acts with no history)
    M1 — sliding window only (recent events, no archive, no reflections)
    M2 — window + archive count (older events acknowledged but not shown)
    M3 — full hierarchical (reflection + important recent + drift anchor) [default]
    """

    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3


@dataclass
class MemoryItem:
    round_id: int
    partner_id: Optional[str]
    event_type: str
    content: str
    outcome: dict[str, Any]
    importance: float = field(default=0.0)
    expires_at_round: Optional[int] = field(default=None)
    """Round after which this belief is no longer surfaced to the LLM.

    None = never expires (default, backwards-compatible).
    Set to round_id + N to create a belief that fades after N rounds.
    Expired items remain in the archive for metric computation but are
    filtered out of get_recent() and _effective_recent so the LLM never
    reasons from stale information.  Mirrors MiroFish's valid_at /
    expired_at pattern on knowledge graph nodes.
    """
    valid_at: Optional[int] = field(default=None)
    """Round when this belief was formed.

    Bookkeeping companion to ``expires_at_round``.  Together they define
    the temporal validity window [valid_at, expires_at_round].  Set
    automatically by ``RoundProcessor._record_memory()``.
    """


class HierarchicalMemory:
    # Compress archive into a reflection whenever it reaches this size.
    _COMPRESS_THRESHOLD = 20

    # Recency half-life: after this many events, weight drops to 50%.
    _RECENCY_HALF_LIFE = 10

    # Importance bonus for social actions (cooperation builds society).
    _SOCIAL_IMPORTANCE = 0.3

    # Batch flush: accumulate this many items before writing to recent/archive.
    # Set to 1 to disable batching (immediate writes, original behaviour).
    _BATCH_FLUSH_THRESHOLD = 5

    # Default time-to-live (in rounds) per event type.  None = never expires.
    # Social/negative experiences linger longer than routine or ephemeral ones.
    # Mirrors MiroFish's valid_at / expired_at pattern on knowledge-graph nodes.
    _DEFAULT_TTL: dict[str, int | None] = {
        "cooperate": 15,  # Social observations fade after 15 rounds
        "work": 10,  # Routine actions fade quickly
        "save": 10,
        "steal": 20,  # Negative experiences linger longer
        "observation": 8,  # NL narrations (kernel feedback) fade after 8 rounds
    }
    _DEFAULT_TTL_FALLBACK: int = 12  # For unknown event types

    @classmethod
    def default_ttl(cls, event_type: str) -> int | None:
        """Return the default time-to-live (in rounds) for an event type.

        Returns the TTL in rounds or ``_DEFAULT_TTL_FALLBACK`` for unknown
        event types.  Used by ``RoundProcessor._record_memory()`` to
        auto-assign ``MemoryItem.expires_at_round``.
        """
        return cls._DEFAULT_TTL.get(event_type, cls._DEFAULT_TTL_FALLBACK)

    def __init__(
        self,
        max_recent: int = 20,
        archive_size: int = 100,
        level: MemoryLevel | int = MemoryLevel.M3,
        persistent_db_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.max_recent = max_recent
        self.archive_size = archive_size
        self.level: MemoryLevel = MemoryLevel(int(level))

        # Optional disk-persistent semantic store (ruflo AgentDB pattern).
        # Stays None — and entirely inert — unless a path is supplied, so
        # default experiments and the M0–M3 ablation are unaffected.
        self.persistent_store = None
        if persistent_db_path:
            from agents.persistent_memory import PersistentMemoryStore

            self.persistent_store = PersistentMemoryStore(persistent_db_path, embedding_model)
        self.recent: list[MemoryItem] = []
        self.archive: list[MemoryItem] = []
        self.reflections: list[str] = []

        # Cache: None means "not yet computed"; "" is a valid cached value.
        self._reflection_cache: Optional[str] = None
        self._cache_dirty: bool = True

        # Pending buffer: items accumulate here until the batch threshold is
        # reached, deferring cache invalidation and archive compression until
        # the full batch is ready.  This mirrors MiroFish's activity-batching
        # strategy (5+ events buffered before pushing to the knowledge graph).
        self._pending_buffer: list[MemoryItem] = []

        # Current simulation round — updated by the kernel each round so that
        # temporal expiry filtering knows what "now" is.
        self._current_round: int = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, item: MemoryItem) -> None:
        """Add a single item immediately (original behaviour, unchanged).

        For bulk round-end writes use ``add_batch()`` which defers cache
        invalidation and compression until all items in the batch are stored.
        """
        if item.importance == 0.0:
            item.importance = self._score_importance(item)
        self.recent.append(item)
        self._cache_dirty = True
        if self.persistent_store is not None:
            self.persistent_store.add(item)

        if len(self.recent) > self.max_recent:
            evicted = self.recent.pop(0)
            self.archive.append(evicted)

            if len(self.archive) > self.archive_size:
                self.archive.pop(0)

            if len(self.archive) % self._COMPRESS_THRESHOLD == 0:
                self._compress_archive()

    def add_batch(self, items: list[MemoryItem]) -> None:
        """Add multiple items in one pass, deferring compression until the end.

        Mirrors MiroFish's activity-batching strategy (buffer N agent events,
        then push to the knowledge graph in one call).  Use this at the end
        of a simulation round to write all agent events for that round in a
        single cache-invalidation cycle rather than one per item.

        Items are first placed in ``_pending_buffer``; once the buffer reaches
        ``_BATCH_FLUSH_THRESHOLD`` (or when ``flush_pending()`` is called) they
        are moved to ``recent`` / ``archive`` and compression runs once.
        """
        for item in items:
            if item.importance == 0.0:
                item.importance = self._score_importance(item)
            self._pending_buffer.append(item)

        if len(self._pending_buffer) >= self._BATCH_FLUSH_THRESHOLD:
            self.flush_pending()

    def flush_pending(self) -> None:
        """Flush all buffered items into recent/archive immediately.

        Call at the end of each simulation round to ensure no events sitting
        in the pending buffer are lost even when the buffer is below the
        batch threshold.
        """
        if not self._pending_buffer:
            return

        if self.persistent_store is not None:
            self.persistent_store.add_batch(list(self._pending_buffer))

        for item in self._pending_buffer:
            self.recent.append(item)

            if len(self.recent) > self.max_recent:
                evicted = self.recent.pop(0)
                self.archive.append(evicted)

                if len(self.archive) > self.archive_size:
                    self.archive.pop(0)

                if len(self.archive) % self._COMPRESS_THRESHOLD == 0:
                    self._compress_archive()

        self._pending_buffer.clear()
        self._cache_dirty = True

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

        all_items = self.archive + self._effective_recent
        self._reflection_cache = self._build_reflection_text(all_items)
        self._cache_dirty = False
        return self._reflection_cache

    @staticmethod
    def _build_reflection_text(items: list[MemoryItem]) -> str:
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
            # Deterministic rounding: int(x + 0.5) avoids Python's
            # banker's-rounding, which can differ across platforms/BLAS and
            # silently change the reflection text (hence the LLM prompt).
            pct = int(100 * w / total_weight + 0.5) if total_weight > 0 else 0
            action_parts.append(f"{action} {pct}%")
        action_summary = ", ".join(action_parts)

        # Partner summary with reciprocation rates from outcome data
        partners: Counter[str] = Counter(m.partner_id for m in items if m.event_type == "cooperate" and m.partner_id)
        partner_summary = ""
        if partners:
            partner_details = []
            for partner, count in partners.most_common(3):
                # Check outcome data for reciprocation info
                partner_items = [m for m in items if m.event_type == "cooperate" and m.partner_id == partner]
                reciprocated = sum(1 for m in partner_items if m.outcome.get("reciprocated") is True)
                total_coop = len(partner_items)
                if reciprocated > 0 or any("reciprocated" in m.outcome for m in partner_items):
                    pct = int(100 * reciprocated / total_coop + 0.5)
                    partner_details.append(f"{partner} (reciprocated {pct}% of the time)")
                else:
                    partner_details.append(partner)
            partner_summary = f" Cooperation partners: {', '.join(partner_details)}."

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

    def advance_round(self, round_id: int) -> int:
        """Advance the memory's sense of "now" and purge expired beliefs.

        Call this at the start of each simulation round (from the kernel or
        round processor).  Returns the count of items that were expired and
        moved to archive so callers can log the expiry event.

        Expired items are moved to archive (not deleted) so they still
        contribute to metric computation and reflection generation.
        """
        self._current_round = round_id
        expired_count = 0
        still_valid: list[MemoryItem] = []
        for item in self.recent:
            if item.expires_at_round is not None and round_id > item.expires_at_round:
                self.archive.append(item)
                if len(self.archive) > self.archive_size:
                    self.archive.pop(0)
                expired_count += 1
            else:
                still_valid.append(item)
        if expired_count:
            self.recent = still_valid
            self._cache_dirty = True
            logger.debug("Memory: expired %d beliefs at round %d", expired_count, round_id)
        return expired_count

    @property
    def _effective_recent(self) -> list[MemoryItem]:
        """Return recent items including any still in the pending buffer.

        Filters out expired items so the LLM never reasons from stale beliefs.
        Pending items are logically in memory even before a flush — exposed
        transparently so batch strategy is invisible to callers.
        """
        now = self._current_round

        def _live(item: MemoryItem) -> bool:
            return item.expires_at_round is None or item.expires_at_round >= now

        return [i for i in self.recent + self._pending_buffer if _live(i)]

    def retrieve(self, query: str | None = None, partner_id: str | None = None, limit: int = 5) -> list[MemoryItem]:
        """Search memory for relevant items by partner_id or keywords."""
        candidates = self._effective_recent + self.archive

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

    def get_recent(self, limit: int = 5) -> list[MemoryItem]:
        """Return the most recent N items (used by prompt builder)."""
        return self._effective_recent[-limit:]

    def get_important_recent(self, limit: int = 5) -> list[MemoryItem]:
        """Return the N most important recent items, sorted chronologically.

        Combines recency (position in list) with the item's importance score
        to decide which memories to surface. This ensures that high-importance
        events (cooperation, large wealth changes) survive even when the
        window is small, preventing social amnesia in long runs.
        """
        eff = self._effective_recent
        if len(eff) <= limit:
            return list(eff)

        half_life = self._RECENCY_HALF_LIFE
        decay = math.log(2) / max(half_life, 1)
        n = len(eff)

        scored = []
        for i, item in enumerate(eff):
            recency_weight = math.exp(decay * (i - n + 1))  # 1.0 for most recent
            combined = 0.6 * recency_weight + 0.4 * item.importance
            scored.append((combined, i, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:limit]
        # Return in chronological order.
        selected.sort(key=lambda x: x[1])
        return [item for _, _, item in selected]

    def get_full_context(self, limit: int = 10) -> list[MemoryItem]:
        """Return a blend of archive and recent events."""
        return (self.archive + self._effective_recent)[-limit:]

    def get_action_distribution(self, weighted: bool = True) -> dict[str, float]:
        """Return the action distribution across all memory items.

        If weighted=True, applies recency weighting. Returns a dict of
        action_type -> proportion (sums to 1.0).
        """
        all_items = self.archive + self._effective_recent
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
