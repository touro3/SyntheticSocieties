"""Shared boilerplate for GPU experiment scripts.

Import this module at the top of any script that runs batched LLM inference
on the experiment cluster. It handles:
  - CUDA device selection
  - Repo root path injection (for editable installs that don't need it, this
    is a no-op)
  - Structured parallel event logging via ParallelLogger
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl


def setup_gpu_env(cuda_devices: str = "0,1") -> None:
    """Set CUDA_VISIBLE_DEVICES and ensure the repo root is on sys.path.

    Call this once at the top of each GPU experiment script, before any
    project imports.

    Args:
        cuda_devices: Comma-separated GPU indices (e.g. "0,1" or "0").
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class ParallelLogger:
    """Lightweight in-memory event collector that flushes to Parquet.

    Usage::

        logger = ParallelLogger()
        logger.log_event({"round_id": 1, "agent_id": "a0", "action": "work"})
        logger.save("experiments/my_exp/events.parquet")
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def log_event(self, event: dict) -> None:
        self.events.append(event)

    def save(self, out_path: str | Path) -> None:
        """Flush all logged events to a Parquet file."""
        if self.events:
            pl.DataFrame(self.events).write_parquet(str(out_path))
