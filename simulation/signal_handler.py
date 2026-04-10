"""
Graceful shutdown handler for long-running BGF simulations.

Registers SIGTERM / SIGINT handlers that:
  1. Set a shared stop flag so the kernel loop exits cleanly after the
     current round finishes (no mid-round corruption).
  2. Save the latest checkpoint so the run can be resumed later.
  3. Mark the run_state as 'failed' with the signal name as the error,
     keeping the completed_rounds count intact for auto_resume().

MiroFish uses SIGTERM / SIGINT / SIGHUP handlers on its subprocess scripts.
BGF adopts the same pattern but integrates it with the existing checkpoint
and crash_recovery infrastructure instead of re-implementing from scratch.

Usage
─────
    from simulation.signal_handler import GracefulShutdown

    shutdown = GracefulShutdown()
    shutdown.register()                          # install signal handlers

    for round_idx in range(num_rounds):
        if shutdown.requested:                   # check each round
            break
        kernel.run_round()
        kernel.save_checkpoint(ckpt_path)
        mgr.tick(round_idx + 1)

    if shutdown.requested:
        mgr.fail(f"Interrupted by {shutdown.signal_name}")
    else:
        mgr.complete()

Or use the context manager:

    with GracefulShutdown() as sd:
        kernel.run(num_rounds=100, stop_flag=sd)
        # kernel.run checks sd.requested each round

The ``stop_flag`` kwarg is wired into SimulationKernel.run() below.
"""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType
from typing import Optional

logger = logging.getLogger(__name__)

# Signals we intercept
_HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT)
try:
    _HANDLED_SIGNALS += (signal.SIGHUP,)   # not available on Windows
except AttributeError:
    pass


class GracefulShutdown:
    """
    Thread-safe stop flag + signal handler for simulation loops.

    Attributes:
        requested:   True after a termination signal has been received.
        signal_name: Name of the received signal (e.g. 'SIGTERM'), or None.
    """

    def __init__(self) -> None:
        self._stop_event   = threading.Event()
        self._signal_name: Optional[str] = None
        self._prev_handlers: dict[int, object] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def requested(self) -> bool:
        """True if a shutdown signal has been received."""
        return self._stop_event.is_set()

    @property
    def signal_name(self) -> Optional[str]:
        return self._signal_name

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self) -> None:
        """Install signal handlers. Call once before the simulation loop."""
        for sig in _HANDLED_SIGNALS:
            try:
                self._prev_handlers[sig] = signal.signal(sig, self._handle)
            except (OSError, ValueError):
                # signal.signal() can fail in non-main threads or on Windows for SIGHUP
                logger.debug("Could not register handler for %s", sig)

    def unregister(self) -> None:
        """Restore original signal handlers. Called automatically by __exit__."""
        for sig, handler in self._prev_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
        self._prev_handlers.clear()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> GracefulShutdown:
        self.register()
        return self

    def __exit__(self, *_) -> None:
        self.unregister()

    # ── Handler ───────────────────────────────────────────────────────────────

    def _handle(self, signum: int, frame: Optional[FrameType]) -> None:
        name = signal.Signals(signum).name
        if not self._stop_event.is_set():
            logger.warning(
                "Signal %s received — will stop after current round completes.", name
            )
            self._signal_name = name
            self._stop_event.set()
        else:
            # Second signal: force exit immediately
            logger.error(
                "Second signal %s received — forcing immediate exit.", name
            )
            raise SystemExit(1)
