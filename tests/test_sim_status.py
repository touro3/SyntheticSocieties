"""Tests for sim_status pure-logic functions.

Covers heartbeat reading, status classification, age formatting, and table
rendering without any TTY output or screen-clearing side effects.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scripts.sim_status import _classify, _read_heartbeat, _age_str, _render_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_hb(path: Path, round_id: int = 1, age_s: float = 0.0, n_agents: int = 4) -> None:
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
    def test_returns_none_when_dir_has_no_heartbeat(self, tmp_path):
        exp = tmp_path / "exp_a"
        exp.mkdir()
        assert _read_heartbeat(exp) is None

    def test_returns_dict_for_valid_heartbeat(self, tmp_path):
        exp = tmp_path / "exp_b"
        exp.mkdir()
        _write_hb(exp / "heartbeat.json", round_id=5)
        hb = _read_heartbeat(exp)
        assert hb is not None
        assert hb["round_id"] == 5

    def test_returns_none_for_corrupt_json(self, tmp_path):
        exp = tmp_path / "exp_c"
        exp.mkdir()
        (exp / "heartbeat.json").write_text("{{bad json")
        assert _read_heartbeat(exp) is None


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_missing_when_hb_is_none(self):
        assert _classify(None, stale_s=300, now=time.time()) == "MISSING"

    def test_ok_for_fresh_heartbeat(self):
        now = time.time()
        hb = {"ts": now - 10}
        assert _classify(hb, stale_s=300, now=now) == "OK"

    def test_stale_between_threshold_and_double(self):
        now = time.time()
        hb = {"ts": now - 400}
        assert _classify(hb, stale_s=300, now=now) == "STALE"

    def test_dead_beyond_double_threshold(self):
        now = time.time()
        hb = {"ts": now - 700}
        assert _classify(hb, stale_s=300, now=now) == "DEAD"


# ---------------------------------------------------------------------------
# _age_str
# ---------------------------------------------------------------------------

class TestAgeStr:
    def test_seconds_for_under_one_minute(self):
        now = time.time()
        result = _age_str(now - 45, now)
        assert "s" in result

    def test_minutes_for_one_to_sixty_minutes(self):
        now = time.time()
        result = _age_str(now - 90, now)
        assert "m" in result

    def test_hours_for_over_one_hour(self):
        now = time.time()
        result = _age_str(now - 7200, now)
        assert "h" in result


# ---------------------------------------------------------------------------
# _render_table
# ---------------------------------------------------------------------------

class TestRenderTable:
    def test_empty_dirs_renders_without_error(self, tmp_path):
        output = _render_table([], stale_s=300, active_only=False)
        assert isinstance(output, str)

    def test_contains_experiment_id(self, tmp_path):
        exp = tmp_path / "my_experiment"
        exp.mkdir()
        _write_hb(exp / "heartbeat.json", round_id=3, age_s=5)
        output = _render_table([exp], stale_s=300, active_only=False)
        assert "my_experiment" in output

    def test_missing_experiment_shown_when_active_only_false(self, tmp_path):
        exp = tmp_path / "ghost_exp"
        exp.mkdir()
        # No heartbeat
        output = _render_table([exp], stale_s=300, active_only=False)
        assert "ghost_exp" in output

    def test_missing_experiment_hidden_when_active_only_true(self, tmp_path):
        exp = tmp_path / "ghost_exp"
        exp.mkdir()
        # No heartbeat — should be excluded
        output = _render_table([exp], stale_s=300, active_only=True)
        assert "ghost_exp" not in output

    def test_fresh_experiment_shown_when_active_only_true(self, tmp_path):
        exp = tmp_path / "live_exp"
        exp.mkdir()
        _write_hb(exp / "heartbeat.json", age_s=5)
        output = _render_table([exp], stale_s=300, active_only=True)
        assert "live_exp" in output

    def test_round_id_appears_in_output(self, tmp_path):
        exp = tmp_path / "exp_r42"
        exp.mkdir()
        _write_hb(exp / "heartbeat.json", round_id=42, age_s=1)
        output = _render_table([exp], stale_s=300, active_only=False)
        assert "42" in output

    def test_summary_counts_all_statuses(self, tmp_path):
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        _write_hb(fresh / "heartbeat.json", age_s=5)

        dead = tmp_path / "dead"
        dead.mkdir()
        _write_hb(dead / "heartbeat.json", age_s=800)

        ghost = tmp_path / "ghost"
        ghost.mkdir()

        output = _render_table([fresh, dead, ghost], stale_s=300, active_only=False)
        # Summary line should mention all four status labels
        assert "OK" in output
        assert "DEAD" in output
        assert "MISSING" in output
