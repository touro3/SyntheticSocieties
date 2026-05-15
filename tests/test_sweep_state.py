"""Tests for the resilient sweep orchestrator state (ruflo autopilot pattern)."""

from __future__ import annotations

from scripts.run_sweep import SweepState


def test_register_is_idempotent_and_persists(tmp_path):
    p = tmp_path / "sweep_state.json"
    s = SweepState(p)
    s.register(["a", "b", "c"])
    s.register(["a", "b", "c"])  # idempotent
    assert set(s.cells) == {"a", "b", "c"}

    reloaded = SweepState(p)
    assert set(reloaded.cells) == {"a", "b", "c"}


def test_pending_excludes_done_but_requeues_running(tmp_path):
    s = SweepState(tmp_path / "state.json")
    s.register(["a", "b", "c"])
    s.mark("a", "done")
    s.mark("b", "running")  # simulates an interrupted cell
    pending = s.pending(["a", "b", "c"])
    assert "a" not in pending
    assert "b" in pending  # running == interrupted == retry
    assert "c" in pending


def test_resume_after_simulated_crash(tmp_path):
    p = tmp_path / "state.json"
    s1 = SweepState(p)
    s1.register(["x", "y"])
    s1.mark("x", "done")
    # process "dies" — new instance from disk
    s2 = SweepState(p)
    assert s2.pending(["x", "y"]) == ["y"]


def test_mark_tracks_attempts_and_summary(tmp_path):
    s = SweepState(tmp_path / "state.json")
    s.register(["a", "b"])
    s.mark("a", "running")
    s.mark("a", "running")
    assert s.cells["a"]["attempts"] == 2
    s.mark("a", "done")
    s.mark("b", "failed", error="boom")
    assert s.summary() == {"done": 1, "failed": 1}
    assert s.cells["b"]["error"] == "boom"
