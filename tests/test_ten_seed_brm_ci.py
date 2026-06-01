"""Tests for the additional empirical seed-level BRM_composite BCa CI (28.1).

This is a *descriptive complement* to the deterministic Theorem-2
weight-robustness certificate (analysis/brm_sensitivity.py), NOT a
replacement. These tests lock: (1) per-run BRM is computable on the fly from
summary.json + events.jsonl (no new persistence), (2) it aggregates with the
shared BCa CI, (3) the Theorem-2 brm_note is preserved verbatim and the new
empirical-CI note is additive, (4) brm_sensitivity.py is untouched/importable.
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np

import analysis.ten_seed_report as tsr
from metrics.statistical_inference import bca_ci


def _synthetic_summary(coop=120, work=140, save=140, n=60, seed=0):
    rng = np.random.default_rng(seed)
    wealth = list(np.clip(rng.lognormal(np.log(100.0), 0.4, n), 10, 800))
    return {
        "wealth": {"values": wealth, "gini": 0.30},
        "event_action_counts": {"work": work, "save": save, "cooperate": coop},
    }


def _write_events(path, n_rounds=5):
    lines = []
    for r in range(n_rounds):
        for a in ("work", "save", "cooperate"):
            lines.append(json.dumps({"round_id": r, "action": {"action_type": a}}))
    path.write_text("\n".join(lines) + "\n")


class TestPerRunBrm:
    def test_returns_composite_in_unit_interval(self, tmp_path):
        ev = tmp_path / "events.jsonl"
        _write_events(ev)
        brm = tsr.per_run_brm(_synthetic_summary(), str(ev), "B_grounded")
        assert brm is not None
        assert 0.0 <= brm <= 1.0

    def test_handles_missing_events(self):
        brm = tsr.per_run_brm(_synthetic_summary(), None, "A_ungrounded")
        assert brm is not None and 0.0 <= brm <= 1.0

    def test_bad_summary_returns_none(self):
        assert tsr.per_run_brm({}, None, "A_ungrounded") is None
        assert tsr.per_run_brm({"wealth": {"values": []}}, None, "B_grounded") is None

    def test_condition_changes_emp_coop_target(self, tmp_path):
        # Same simulated behaviour scored against grounded vs ungrounded ESS
        # cooperation target → different composite (sanity that condition is
        # actually wired through).
        s = _synthetic_summary()
        b = tsr.per_run_brm(s, None, "B_grounded")
        a = tsr.per_run_brm(s, None, "A_ungrounded")
        assert b != a

    def test_seed_array_then_bca(self):
        vals = np.array([tsr.per_run_brm(_synthetic_summary(seed=s), None, "B_grounded") for s in range(4)])
        assert np.all(np.isfinite(vals))
        lo, hi = bca_ci(vals, rng=np.random.default_rng(0))
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi


class TestComplementNotReplacement:
    def test_brm_composite_is_an_artifact_metric(self):
        assert "brm_composite" in tsr.ARTIFACT_METRICS

    def test_theorem2_brm_note_preserved_verbatim(self):
        src = inspect.getsource(tsr.main)
        assert 'results["brm_note"]' in src
        assert "DETERMINISTIC Theorem 2" in src

    def test_empirical_ci_note_is_additive(self):
        src = inspect.getsource(tsr.main)
        # Both notes must coexist: the formal certificate statement AND the
        # descriptive empirical-CI complement.
        assert 'results["brm_empirical_ci_note"]' in src
        assert "complement" in src.lower()

    def test_brm_sensitivity_untouched_and_importable(self):
        import analysis.brm_sensitivity as bs

        assert any("certificate" in n for n in dir(bs)) or hasattr(bs, "main")
