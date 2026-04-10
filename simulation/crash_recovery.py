"""
Crash recovery and resumable simulation management.

Writes a ``run_state.json`` file per experiment directory that tracks the
simulation lifecycle (pending → running → complete / failed).  When a GPU
run is interrupted mid-way, ``auto_resume()`` detects the incomplete state
and returns the correct ``start_round`` so the kernel can pick up exactly
where it left off.

Integration with the existing kernel
─────────────────────────────────────
The ``SimulationKernel`` already has ``save_checkpoint()`` / ``load_checkpoint()``
(agents states + round_id).  This module adds the higher-level orchestration
layer:

  1. ``RunStateManager.start()``  — write status=running before the first round
  2. kernel.run() loop            — checkpoint written each round (existing code)
  3. ``RunStateManager.complete()`` — write status=complete after last round
  4. On crash: status stays "running" + checkpoint.json is the last good round

Resuming a crashed run
───────────────────────
    from simulation.crash_recovery import auto_resume

    start_round, checkpoint_path = auto_resume("experiments/exp_001")
    if checkpoint_path:
        kernel.load_checkpoint(checkpoint_path)
    kernel.run(num_rounds=100, start_round=start_round)

Usage
─────
    from simulation.crash_recovery import RunStateManager

    mgr = RunStateManager("experiments/exp_001")
    mgr.start(total_rounds=100, experiment_id="exp_001")
    try:
        kernel.run(num_rounds=100, start_round=mgr.resume_round)
        mgr.complete()
    except Exception as exc:
        mgr.fail(str(exc))
        raise
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STATE_FILE   = "run_state.json"
_CHECKPOINT   = "checkpoint.json"
_GRAPH_EVENTS = "events.jsonl"   # used for GraphRAG replay on resume


# ── RunState dataclass ────────────────────────────────────────────────────────

@dataclass
class RunState:
    """Mutable snapshot of a simulation run's lifecycle state."""

    experiment_id:  str
    status:         str          # pending | running | complete | failed
    total_rounds:   int
    completed_rounds: int = 0
    started_at:     float = field(default_factory=time.time)
    updated_at:     float = field(default_factory=time.time)
    finished_at:    Optional[float] = None
    error_message:  Optional[str] = None

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_resumable(self) -> bool:
        """True if the run started but did not complete cleanly."""
        return self.status == "running" and self.completed_rounds > 0

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def resume_round(self) -> int:
        """Round to pass as start_round when resuming."""
        return self.completed_rounds if (self.is_resumable or self.is_failed) else 0


# ── RunStateManager ───────────────────────────────────────────────────────────

class RunStateManager:
    """
    Manages the run_state.json lifecycle for one experiment directory.

    Args:
        exp_dir: Path to the experiment directory (e.g. 'experiments/exp_001').
    """

    def __init__(self, exp_dir: str | Path):
        self._dir   = Path(exp_dir)
        self._path  = self._dir / _STATE_FILE
        self._state: Optional[RunState] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> Optional[RunState]:
        return self._state

    @property
    def resume_round(self) -> int:
        """Convenience: start_round to use when calling kernel.run()."""
        return self._state.resume_round if self._state else 0

    @property
    def checkpoint_path(self) -> Optional[Path]:
        p = self._dir / _CHECKPOINT
        return p if p.exists() else None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, total_rounds: int, experiment_id: str) -> RunState:
        """Create (or overwrite) run_state.json with status=running."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state = RunState(
            experiment_id=experiment_id,
            status="running",
            total_rounds=total_rounds,
            completed_rounds=0,
        )
        self._write()
        logger.info("RunState started: %s (total_rounds=%d)", experiment_id, total_rounds)
        return self._state

    def tick(self, completed_rounds: int) -> None:
        """Update completed_rounds after each round (call from within kernel loop)."""
        if self._state is None:
            return
        self._state.completed_rounds = completed_rounds
        self._state.updated_at = time.time()
        self._write()

    def complete(self) -> None:
        """Mark the run as successfully completed."""
        if self._state is None:
            return
        self._state.status       = "complete"
        self._state.finished_at  = time.time()
        self._state.updated_at   = time.time()
        self._write()
        logger.info("RunState complete: %s", self._state.experiment_id)

    def fail(self, error_message: str = "") -> None:
        """Mark the run as failed (preserves completed_rounds for resume)."""
        if self._state is None:
            return
        self._state.status        = "failed"
        self._state.error_message = error_message[:500]
        self._state.updated_at    = time.time()
        self._write()
        logger.error(
            "RunState failed: %s at round %d — %s",
            self._state.experiment_id, self._state.completed_rounds, error_message,
        )

    # ── Load existing state ───────────────────────────────────────────────────

    def load(self) -> Optional[RunState]:
        """Read run_state.json from disk. Returns None if file doesn't exist."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._state = RunState.from_dict(data)
            return self._state
        except Exception as exc:
            logger.warning("Could not parse run_state.json (%s): %s", self._path, exc)
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self) -> None:
        if self._state is None:
            return
        try:
            self._path.write_text(
                json.dumps(self._state.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write run_state.json: %s", exc)


# ── auto_resume() ─────────────────────────────────────────────────────────────

def auto_resume(
    exp_dir: str | Path,
    total_rounds: int,
    experiment_id: str,
) -> tuple[int, Optional[Path], RunStateManager]:
    """
    Detect whether a previous run is resumable and return the start round.

    This is the recommended high-level entry point.  It handles three cases:

      1. No run_state.json   → fresh start (start_round=0)
      2. status=complete     → already done; start_round=total_rounds (no-op)
      3. status=running/failed with checkpoint → resume (start_round=N)

    In case 3, the GraphRAG is also replayed from the saved events.jsonl so
    the social graph is consistent with the checkpoint state.

    Args:
        exp_dir:       Experiment directory.
        total_rounds:  Total rounds planned for this run.
        experiment_id: Experiment identifier string.

    Returns:
        (start_round, checkpoint_path, mgr)
        - start_round:      Pass to ``kernel.run(num_rounds=..., start_round=...)``.
        - checkpoint_path:  Pass to ``kernel.load_checkpoint()`` if not None.
        - mgr:              ``RunStateManager`` — call mgr.tick() / mgr.complete() / mgr.fail().
    """
    mgr = RunStateManager(exp_dir)
    existing = mgr.load()

    if existing is None:
        # Fresh start
        mgr.start(total_rounds=total_rounds, experiment_id=experiment_id)
        logger.info("auto_resume: fresh start for %s", experiment_id)
        return 0, None, mgr

    if existing.is_complete:
        # Already done — caller can skip the run
        logger.info(
            "auto_resume: %s already complete (%d rounds). Nothing to do.",
            experiment_id, existing.total_rounds,
        )
        return existing.total_rounds, None, mgr

    if existing.is_resumable or existing.is_failed:
        start = existing.resume_round
        ckpt  = mgr.checkpoint_path
        logger.info(
            "auto_resume: resuming %s from round %d (status was '%s'). Checkpoint: %s",
            experiment_id, start, existing.status, ckpt,
        )
        # Re-open the state as running so we track the resumed run
        existing.status      = "running"
        existing.updated_at  = time.time()
        existing.error_message = None
        mgr._state = existing
        mgr._write()
        return start, ckpt, mgr

    # Fallback: unexpected status — start fresh
    logger.warning(
        "auto_resume: unexpected status '%s' for %s — starting fresh.",
        existing.status, experiment_id,
    )
    mgr.start(total_rounds=total_rounds, experiment_id=experiment_id)
    return 0, None, mgr


# ── Utility: replay GraphRAG from saved events ────────────────────────────────

def replay_graph_rag(exp_dir: str | Path, graph_rag: object) -> None:
    """Rebuild a GraphRAG instance from saved events.jsonl.

    Call this after ``kernel.load_checkpoint()`` to ensure the social graph
    matches the checkpoint state before resuming inference.

    Args:
        exp_dir:   Experiment directory containing events.jsonl.
        graph_rag: A ``GraphRAG`` instance (from ``decision.graph_rag``).
    """
    events_path = Path(exp_dir) / _GRAPH_EVENTS
    if not events_path.exists():
        logger.info("replay_graph_rag: no events.jsonl found at %s — skipping.", events_path)
        return
    if hasattr(graph_rag, "build_from_events"):
        graph_rag.build_from_events(events_path)
        logger.info("replay_graph_rag: rebuilt from %s", events_path)
    else:
        logger.warning("replay_graph_rag: provided object has no build_from_events() method.")


# ── scan_incomplete_runs() ────────────────────────────────────────────────────

def scan_incomplete_runs(experiments_root: str | Path = "experiments") -> list[dict]:
    """Scan an experiments directory and return all incomplete (resumable) runs.

    Useful for a pre-flight check before launching a batch of GPU jobs.

    Returns:
        List of dicts with keys: experiment_id, exp_dir, completed_rounds,
        total_rounds, status, updated_at.
    """
    root = Path(experiments_root)
    if not root.exists():
        return []

    results = []
    for state_file in sorted(root.rglob(_STATE_FILE)):
        try:
            data  = json.loads(state_file.read_text(encoding="utf-8"))
            state = RunState.from_dict(data)
            if not state.is_complete:
                results.append({
                    "experiment_id":    state.experiment_id,
                    "exp_dir":          str(state_file.parent),
                    "completed_rounds": state.completed_rounds,
                    "total_rounds":     state.total_rounds,
                    "status":           state.status,
                    "updated_at":       state.updated_at,
                })
        except Exception:
            continue

    return results
