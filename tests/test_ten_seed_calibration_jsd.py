"""Tests for the calibration-JSD headline metric (Phase 28.1).

Calibration was previously the one headline metric with NO seed-level CI
(metrics/calibration.py is a 3-seed cal/eval split with explicitly zero
uncertainty bounds). This adds `calibration_jsd` — JSD between the simulated
end-state wealth distribution and the ESS income reference, both min-max
normalized so it is a scale-invariant distribution-*shape* divergence —
aggregated across the confirmatory seeds with the shared BCa CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics.calibration import calibration_jsd, ess_wealth_reference
from metrics.statistical_inference import bca_ci


class TestEssWealthReference:
    def test_deterministic(self):
        a = ess_wealth_reference()
        b = ess_wealth_reference()
        assert np.array_equal(a, b)

    def test_in_income_decile_range_and_finite(self):
        ref = ess_wealth_reference()
        assert ref.size > 0
        assert np.all(np.isfinite(ref))
        # Household net income decile is bounded [1, 10] in the ESS extract.
        assert ref.min() >= 1.0 - 1e-9
        assert ref.max() <= 10.0 + 1e-9

    def test_reflects_ess_central_tendency(self):
        # ESS income_decile mean ≈ 5.04, median 5.0 — reconstruction should
        # land in the right neighbourhood (not a degenerate constant).
        ref = ess_wealth_reference()
        assert 4.0 < float(np.mean(ref)) < 6.0
        assert ref.std() > 0.5


class TestCalibrationJsd:
    def test_identical_distribution_is_zero(self):
        x = np.array([10.0, 25.0, 40.0, 55.0, 70.0, 90.0, 120.0, 200.0])
        assert calibration_jsd(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_shapes_are_high(self):
        # Independent min-max normalization makes this a *shape* divergence
        # (location/scale are factored out — that is the intended, tested
        # scale-invariance). So the contrast must differ in SHAPE: a
        # right-skewed vs a left-skewed distribution (mirror images).
        rng = np.random.default_rng(7)
        right_skew = rng.exponential(1.0, size=600)
        left_skew = 12.0 - rng.exponential(1.0, size=600)
        jsd = calibration_jsd(right_skew, left_skew)
        assert jsd > 0.3  # base-2 JSD is bounded by 1.0

    def test_scale_invariant(self):
        # THE regression guard for the wealth-vs-decile scale trap:
        # multiplying simulated wealth by a constant must not change the
        # (shape) calibration JSD, because each array is min-max normalized.
        rng = np.random.default_rng(0)
        sim = rng.lognormal(3.0, 0.8, size=200)
        ess = ess_wealth_reference()
        base = calibration_jsd(sim, ess)
        scaled = calibration_jsd(sim * 1000.0, ess)
        assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)

    def test_constant_sim_does_not_crash(self):
        ess = ess_wealth_reference()
        jsd = calibration_jsd(np.full(50, 7.5), ess)
        assert math.isfinite(jsd)

    def test_per_seed_jsd_then_bca(self):
        ess = ess_wealth_reference()
        rng = np.random.default_rng(1)
        per_seed = np.array([calibration_jsd(rng.lognormal(3.0, 0.8, size=150), ess) for _ in range(5)])
        assert per_seed.shape == (5,)
        assert np.all(np.isfinite(per_seed))
        lo, hi = bca_ci(per_seed, rng=np.random.default_rng(1))
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi


class TestPressPlayGuardPreserved:
    def test_load_per_run_artifacts_no_registry_returns_empty(self, monkeypatch, tmp_path):
        import analysis.ten_seed_report as tsr

        monkeypatch.setattr(tsr, "REGISTRY", tmp_path / "does_not_exist.parquet")
        assert tsr.load_per_run_artifacts() == {}

    def test_load_per_run_artifacts_never_raises(self, monkeypatch, tmp_path):
        # Even a malformed registry path must degrade to {} so the
        # press-play awaiting_runs path is never broken by artifact joins.
        import analysis.ten_seed_report as tsr

        monkeypatch.setattr(tsr, "REGISTRY", tmp_path / "nope.parquet")
        out = tsr.load_per_run_artifacts()
        assert isinstance(out, dict) and out == {}
