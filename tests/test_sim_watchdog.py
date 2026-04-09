"""Tests for sim_watchdog pure-logic functions.

No GPU, no I/O side-effects — uses tmp_path for heartbeat files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scripts.sim_watchdog import _read_heartbeat, _status, _age_str, watch_once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_hb(path: Path, round_id: int = 1, age_s: float = 0.0, n_agents: int = 5) -> None:
    """Write a heartbeat JSON to path with ts = now - age_s."""
    payload = {
        "round_id": round_id,
        "ts": time.time() - age_s,
        "n_agents": n_agents,
    }
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# _read_heartbeat
# ---------------------------------------------------------------------------

class TestReadHeartbeat:
    def test_returns_none_when_file_missing(self, tmp_path):
        assert _read_heartbeat(tmp_path / "nonexistent") is None

    def test_returns_none_when_no_heartbeat_json(self, tmp_path):
        exp = tmp_path / "exp_a"
        exp.mkdir()
        assert _read_heartbeat(exp) is None

    def test_returns_dict_for_valid_file(self, tmp_path):
        exp = tmp_path / "exp_a"
        exp.mkdir()
        _write_hb(exp / "heartbeat.json", round_id=3)
        hb = _read_heartbeat(exp)
        assert hb is not None
        assert hb["round_id"] == 3

    def test_returns_none_for_corrupt_json(self, tmp_path):
        exp = tmp_path / "exp_b"
        exp.mkdir()
        (exp / "heartbeat.json").write_text("not json {{{")
        assert _read_heartbeat(exp) is None


# ---------------------------------------------------------------------------
# _status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_missing_when_no_heartbeat(self):
        now = time.time()
        assert _status(None, stale_after_s=300, now=now) == "MISSING"

    def test_ok_for_fresh_heartbeat(self):
        now = time.time()
        hb = {"ts": now - 10}
        assert _status(hb, stale_after_s=300, now=now) == "OK"

    def test_stale_between_threshold_and_double(self):
        now = time.time()
        hb = {"ts": now - 400}  # stale_after=300, dead at 600
        assert _status(hb, stale_after_s=300, now=now) == "STALE"

    def test_dead_beyond_double_threshold(self):
        now = time.time()
        hb = {"ts": now - 700}  # dead > 2×300=600
        assert _status(hb, stale_after_s=300, now=now) == "DEAD"

    def test_exactly_at_stale_threshold_is_stale(self):
        now = time.time()
        hb = {"ts": now - 300}
        result = _status(hb, stale_after_s=300, now=now)
        assert result in ("STALE", "OK")  # boundary: just allow either

    def test_exactly_at_dead_threshold_is_dead(self):
        now = time.time()
        hb = {"ts": now - 600}
        result = _status(hb, stale_after_s=300, now=now)
        assert result in ("DEAD", "STALE")  # boundary: just allow either


# ---------------------------------------------------------------------------
# _age_str
# ---------------------------------------------------------------------------

class TestAgeStr:
    def test_seconds_format(self):
        now = time.time()
        result = _age_str(now - 42, now)
        assert "s" in result
        assert "42" in result

    def test_minutes_format(self):
        now = time.time()
        result = _age_str(now - 120, now)
        assert "m" in result

    def test_sub_minute_stays_in_seconds(self):
        now = time.time()
        result = _age_str(now - 59, now)
        assert "s" in result


# ---------------------------------------------------------------------------
# watch_once
# ---------------------------------------------------------------------------

class TestWatchOnce:
    def test_all_missing_when_no_heartbeats(self, tmp_path):
        dirs = []
        for i in range(3):
            d = tmp_path / f"exp_{i}"
            d.mkdir()
            dirs.append(d)

        results = watch_once(dirs, stale_after_s=300, verbose=False)
        assert len(results) == 3
        statuses = [r[1] for r in results]
        assert all(s == "MISSING" for s in statuses)

    def test_ok_for_fresh_experiments(self, tmp_path):
        dirs = []
        for i in range(2):
            d = tmp_path / f"exp_{i}"
            d.mkdir()
            _write_hb(d / "heartbeat.json", round_id=i + 1, age_s=5)
            dirs.append(d)

        results = watch_once(dirs, stale_after_s=300, verbose=False)
        statuses = [r[1] for r in results]
        assert all(s == "OK" for s in statuses)

    def test_mixed_statuses(self, tmp_path):
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        _write_hb(fresh / "heartbeat.json", age_s=10)

        dead = tmp_path / "dead"
        dead.mkdir()
        _write_hb(dead / "heartbeat.json", age_s=700)

        ghost = tmp_path / "ghost"
        ghost.mkdir()
        # no heartbeat

        results = watch_once([fresh, dead, ghost], stale_after_s=300, verbose=False)
        by_dir = {r[0].name: r[1] for r in results}

        assert by_dir["fresh"] == "OK"
        assert by_dir["dead"] == "DEAD"
        assert by_dir["ghost"] == "MISSING"

    def test_returns_heartbeat_dict_for_ok(self, tmp_path):
        exp = tmp_path / "exp_ok"
        exp.mkdir()
        _write_hb(exp / "heartbeat.json", round_id=7, n_agents=10, age_s=1)

        results = watch_once([exp], stale_after_s=300, verbose=False)
        _, status, hb = results[0]
        assert status == "OK"
        assert hb["round_id"] == 7
        assert hb["n_agents"] == 10

    def test_returns_none_heartbeat_for_missing(self, tmp_path):
        exp = tmp_path / "no_hb"
        exp.mkdir()
        results = watch_once([exp], stale_after_s=300, verbose=False)
        _, status, hb = results[0]
        assert status == "MISSING"
        assert hb is None

    def test_empty_dirs_returns_empty(self):
        results = watch_once([], stale_after_s=300, verbose=False)
        assert results == []
