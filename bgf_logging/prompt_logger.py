"""Prompt and output logger for LLM decision auditing.

Stores every prompt + LLM response as JSONL for reproducibility and analysis.
Supports two OOM safeguards:

  * **File-size rotation** (``max_bytes``, default 100 MB): when the current
    file exceeds the limit the logger rotates to a numbered shard
    (prompts.0001.jsonl, prompts.0002.jsonl, …).  The active file is always
    named ``prompts.jsonl``.

  * **Sampling** (``sample_rate``, default 1.0 = log everything): set to a
    value in (0, 1] to log only that fraction of records.  Useful for large
    runs (e.g. 500 agents × 10 000 rounds) where full logging would produce
    tens of GB.  Records are selected with a round-robin counter so every Nth
    record is logged — deterministic and evenly spread.

Performance: a persistent file handle is kept open for the lifetime of the
logger (line-buffered).  This eliminates the open/close syscall overhead on
every log() call — at 100 agents × 30 rounds that saves ~3 000 syscalls per
experiment.

Examples::

    # Default: log everything, rotate at 100 MB
    pl = PromptLogger("experiments/exp1/prompts.jsonl")

    # Large run: log 10% of prompts, rotate at 50 MB
    pl = PromptLogger("experiments/exp1/prompts.jsonl", sample_rate=0.1, max_bytes=50*1024*1024)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import IO, Optional

logger = logging.getLogger(__name__)

# 100 MB per shard — prompts are larger than events (~2 KB each) so a tighter
# default prevents individual shards from becoming unwieldy.
_DEFAULT_MAX_BYTES = 100 * 1024 * 1024


class PromptLogger:
    """Log prompts and LLM outputs to a JSONL file with rotation and sampling."""

    def __init__(
        self,
        output_path: str | Path,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        sample_rate: float = 1.0,
    ):
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be in (0, 1]; got {sample_rate}")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._sample_rate = sample_rate

        # Round-robin sampling: log every ceil(1/sample_rate)-th record.
        self._sample_every: int = max(1, math.ceil(1.0 / sample_rate))
        self._total_calls = 0  # total log() calls (including skipped)
        self._count = 0  # records actually written
        self._rotation_count = 0
        self._bytes_written = 0
        self._fh: Optional[IO[str]] = None

        # Resume byte tracking if the file already exists.
        if self.output_path.exists():
            self._bytes_written = self.output_path.stat().st_size

        self._open_handle()

        if sample_rate < 1.0:
            logger.info(
                "PromptLogger: sampling 1-in-%d records (sample_rate=%.2f)",
                self._sample_every,
                sample_rate,
            )

    # ── Handle management ─────────────────────────────────────────────────────

    def _open_handle(self) -> None:
        """Open (or reopen) the persistent append handle, line-buffered."""
        self._fh = self.output_path.open("a", encoding="utf-8", buffering=1)

    # ── Rotation ──────────────────────────────────────────────────────────────

    def _rotate(self) -> None:
        """Flush, close, rename the current file, then open a fresh handle."""
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

        self._rotation_count += 1
        shard_path = self.output_path.with_suffix(f".{self._rotation_count:04d}.jsonl")
        self.output_path.rename(shard_path)
        self._bytes_written = 0
        logger.info(
            "PromptLogger: rotated to shard %s (shard %d, %d records written so far)",
            shard_path.name,
            self._rotation_count,
            self._count,
        )
        self._open_handle()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(
        self,
        round_id: int,
        agent_id: str,
        prompt: str,
        raw_output: str,
        parsed_action: Optional[dict],
        latency_ms: float,
        parse_metadata: Optional[dict] = None,
        rag_context: Optional[dict] = None,
    ) -> None:
        """Append one prompt/output record to the JSONL file.

        Args:
            rag_context: Optional dict with RAG presence flags, e.g.
                {"sql_rag_present": True, "graph_rag_present": False}.
                Enables post-hoc verification that grounding was active.
        """
        self._total_calls += 1

        # Round-robin sampling: skip records not on the sample cadence.
        if self._total_calls % self._sample_every != 1:
            return

        record = {
            "round_id": round_id,
            "agent_id": agent_id,
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed_action": parsed_action,
            "latency_ms": round(latency_ms, 2),
            "parse_metadata": {
                k: v
                for k, v in (parse_metadata or {}).items()
                if k != "raw_text"  # avoid duplication with prompt field
            },
            "rag_context": rag_context,
        }

        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        # Rotate *before* writing if the next record would exceed the limit.
        if self._bytes_written + line_bytes > self._max_bytes:
            self._rotate()

        self._fh.write(line)
        self._bytes_written += line_bytes
        self._count += 1

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of records actually written (after sampling)."""
        return self._count

    @property
    def total_calls(self) -> int:
        """Total log() calls including skipped (sampled-out) records."""
        return self._total_calls

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the file handle. Safe to call multiple times."""
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> PromptLogger:
        return self

    def __exit__(self, *_) -> None:
        self.close()
