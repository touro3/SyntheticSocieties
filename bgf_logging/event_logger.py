"""Event logger with file-size-based rotation.

Writes simulation events to a JSONL file and rotates to a numbered shard
(events.0001.jsonl, events.0002.jsonl, …) when the current file exceeds
``max_bytes``.  The active file is always named ``events.jsonl`` so existing
consumers need no changes when rotation never triggers.

Default: 200 MB per shard.  A 500-agent × 10 000-round run produces ~2.5 GB
of events; with the default limit that creates ≤ 13 shards, each independently
readable.

Performance: a persistent file handle is kept open for the lifetime of the
logger (line-buffered).  This eliminates the open/close syscall overhead on
every log_event() call — at 100 agents × 30 rounds that saves ~3 000 syscalls
per experiment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, Optional

logger = logging.getLogger(__name__)

# 200 MB default per shard — large enough that most research runs never rotate,
# yet prevents any single file from exceeding typical inode/FS limits.
_DEFAULT_MAX_BYTES = 200 * 1024 * 1024


class EventLogger:
    """Append-only JSONL event logger with size-capped file rotation."""

    def __init__(
        self,
        output_path: str | Path,
        overwrite: bool = False,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._rotation_count = 0
        self._bytes_written = 0
        self._fh: Optional[IO[str]] = None

        if overwrite and self.output_path.exists():
            self.output_path.unlink()

        # Resume byte tracking if the file already exists (e.g., after restart).
        if self.output_path.exists():
            self._bytes_written = self.output_path.stat().st_size

        self._open_handle()

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
            "EventLogger: rotated to shard %s (shard %d)",
            shard_path.name,
            self._rotation_count,
        )
        self._open_handle()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_event(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        # Rotate *before* writing if the next record would exceed the limit.
        if self._bytes_written + line_bytes > self._max_bytes:
            self._rotate()

        self._fh.write(line)
        self._bytes_written += line_bytes

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the file handle. Safe to call multiple times."""
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *_) -> None:
        self.close()
