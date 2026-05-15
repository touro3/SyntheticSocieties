"""Memory-activation diagnostics (Phase 3, audit response — "myopic agents").

Research motivation
-------------------
A reviewer can object that the hierarchical memory is *architecturally present
but behaviorally inert*: agents may never actually condition on retrieved
memories, so any realism is not memory-driven. We need direct evidence that
memory is **activated** — retrieved and surfaced into the prompt the model
actually reasons over.

This module measures, per agent per round, two quantities:

  * **retrieval frequency** — how many memory items the prompt builder pulled
    (``get_recent`` / ``get_important_recent``) at the configured window/level;
  * **citation rate** — what fraction of those retrieved items actually appear
    in the rendered memory block (round id + partner + event type), i.e. how
    much of what was retrieved survived into the text the LLM sees.

Design (intentionally non-invasive)
-----------------------------------
We do **not** modify ``decision.prompt_builder.build_memory_block`` or the
kernel loop. Instead :func:`build_memory_block_with_diagnostics` wraps the
existing builder: it calls it unchanged, independently re-derives what the
builder would have retrieved for the same ``memory``/``window``/``level``, and
scores citation against the returned text. Default behavior is unchanged and
zero-cost unless a logger is passed (gated).

The JSONL output is a thesis-methodology artifact: it records the full
per-round retrieval/citation trajectory, not just an endpoint summary.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import IO, Optional

from agents.memory import HierarchicalMemory, MemoryLevel

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 100 * 1024 * 1024


class MemoryDiagnosticsLogger:
    """Append memory-activation records to a rotating JSONL file.

    Mirrors :class:`bgf_logging.prompt_logger.PromptLogger`'s rotation and
    round-robin sampling contract so operational behavior is familiar.
    """

    def __init__(
        self,
        output_path: str | Path,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        sample_rate: float = 1.0,
    ) -> None:
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be in (0, 1]; got {sample_rate}")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._sample_every = max(1, math.ceil(1.0 / sample_rate))
        self._total_calls = 0
        self._count = 0
        self._rotation_count = 0
        self._bytes_written = self.output_path.stat().st_size if self.output_path.exists() else 0
        self._fh: Optional[IO[str]] = self.output_path.open("a", encoding="utf-8", buffering=1)

    def _rotate(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
        self._rotation_count += 1
        shard = self.output_path.with_suffix(f".{self._rotation_count:04d}.jsonl")
        self.output_path.rename(shard)
        self._bytes_written = 0
        self._fh = self.output_path.open("a", encoding="utf-8", buffering=1)

    def log(self, record: dict) -> None:
        """Append one diagnostics record (round-robin sampled)."""
        self._total_calls += 1
        # sample_rate==1.0 ⇒ sample_every==1 ⇒ log every call; otherwise keep
        # one in every `_sample_every` records (round-robin).
        if self._sample_every > 1 and self._total_calls % self._sample_every != 1:
            return
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if self._bytes_written + line_bytes > self._max_bytes:
            self._rotate()
        assert self._fh is not None
        self._fh.write(line)
        self._bytes_written += line_bytes
        self._count += 1

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def _retrieved_for(memory: HierarchicalMemory, window: int, level: MemoryLevel) -> list:
    """Re-derive exactly what build_memory_block would retrieve for this level.

    Mirrors the retrieval branch selection in
    ``decision.prompt_builder.build_memory_block`` (M0 none, M1/M2 recent,
    M3 importance-weighted) without rendering anything.
    """
    if level == MemoryLevel.M0:
        return []
    if level in (MemoryLevel.M1, MemoryLevel.M2):
        return memory.get_recent(window) if hasattr(memory, "get_recent") else []
    # M3
    if hasattr(memory, "get_important_recent"):
        return memory.get_important_recent(window)
    if hasattr(memory, "get_recent"):
        return memory.get_recent(window)
    return []


def memory_activation_record(
    memory: HierarchicalMemory,
    window: int,
    level: MemoryLevel,
    rendered_block: str,
    *,
    round_id: int = -1,
    agent_id: str = "",
) -> dict:
    """Build a per-agent/round memory-activation diagnostics record.

    ``citation`` counts a retrieved item as *cited* iff the rendered memory
    block contains its ``"Round {round_id}: you chose '{event_type}'"`` stem
    (and its partner id when present) — exactly the surface form the prompt
    builder emits — so the rate reflects what the LLM can actually read.
    """
    retrieved = _retrieved_for(memory, window, level)
    cited = 0
    for item in retrieved:
        stem = f"Round {item.round_id}: you chose '{item.event_type}'"
        ok = stem in rendered_block
        if ok and item.partner_id:
            ok = item.partner_id in rendered_block
        cited += int(ok)
    n = len(retrieved)
    return {
        "agent_id": agent_id,
        "round_id": round_id,
        "memory_level": int(level),
        "window": window,
        "retrieval_count": n,
        "citation_count": cited,
        "citation_rate": (cited / n) if n else 0.0,
        "recent_pool_size": len(getattr(memory, "_effective_recent", [])),
        "archive_size": len(getattr(memory, "archive", [])),
        "retrieved_refs": [
            {"round_id": it.round_id, "event_type": it.event_type, "partner_id": it.partner_id} for it in retrieved
        ],
    }


def build_memory_block_with_diagnostics(
    memory: HierarchicalMemory,
    window: int = 5,
    profile=None,
    level=None,
    *,
    logger_sink: Optional[MemoryDiagnosticsLogger] = None,
    round_id: int = -1,
    agent_id: str = "",
) -> str:
    """Drop-in wrapper around ``build_memory_block`` that also emits diagnostics.

    Returns the identical string the unwrapped builder produces. If
    ``logger_sink`` is None this is a pure pass-through (zero behavioral
    change), satisfying the backward-compatibility constraint.
    """
    from decision.prompt_builder import build_memory_block

    text = build_memory_block(memory, window=window, profile=profile, level=level)

    if logger_sink is not None:
        if level is None:
            level = getattr(memory, "level", MemoryLevel.M3)
        rec = memory_activation_record(
            memory,
            window,
            MemoryLevel(int(level)),
            text,
            round_id=round_id,
            agent_id=agent_id,
        )
        logger_sink.log(rec)

    return text
