"""Unit tests for the shared BCa bootstrap CI.

`bca_ci` backs every headline seed-level 95% CI in the ten-seed confirmatory
pipeline (cooperation_rate, wealth_gini, b_rlhf, calibration_jsd, brm_composite),
yet was previously untested. These tests lock its contract before it is
extracted from analysis/ten_seed_report.py into metrics/statistical_inference.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics.statistical_inference import bca_ci


class TestBcaCI:
    def test_known_normal_coverage(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=5.0, scale=2.0, size=40)
        lo, hi = bca_ci(x, rng=np.random.default_rng(0))
        assert lo < x.mean() < hi
        # True mean 5.0 should sit inside a 95% CI for n=40 from N(5,2).
        assert lo < 5.0 < hi

    def test_deterministic_with_explicit_rng(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        r1 = bca_ci(x, rng=np.random.default_rng(7))
        r2 = bca_ci(x, rng=np.random.default_rng(7))
        assert r1 == r2

    def test_default_rng_is_seed_42_stable(self):
        # rng=None must mean a *fresh* default_rng(42) each call (paper seed
        # convention), so back-to-back default calls are identical.
        x = [0.2, 0.5, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7]
        assert bca_ci(x) == bca_ci(x)

    def test_degenerate_n1_returns_nan(self):
        lo, hi = bca_ci([3.0])
        assert math.isnan(lo) and math.isnan(hi)

    def test_empty_returns_nan(self):
        lo, hi = bca_ci([])
        assert math.isnan(lo) and math.isnan(hi)

    def test_n2_does_not_crash(self):
        lo, hi = bca_ci([1.0, 3.0], rng=np.random.default_rng(1))
        assert math.isfinite(lo) and math.isfinite(hi)
        assert lo <= hi

    def test_lo_le_hi(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 1, size=25)
        lo, hi = bca_ci(x, rng=np.random.default_rng(2))
        assert lo <= hi

    def test_constant_input_zero_width(self):
        # All identical → bootstrap distribution is the constant; CI collapses.
        lo, hi = bca_ci([5.0] * 30, rng=np.random.default_rng(3))
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_alpha_widens_interval(self):
        x = list(np.random.default_rng(4).normal(0, 1, size=30))
        lo95, hi95 = bca_ci(x, alpha=0.05, rng=np.random.default_rng(99))
        lo99, hi99 = bca_ci(x, alpha=0.01, rng=np.random.default_rng(99))
        assert (hi99 - lo99) >= (hi95 - lo95)
        assert lo99 <= lo95 and hi99 >= hi95

    def test_bca_vs_percentile_sanity_symmetric(self):
        # On a large symmetric sample BCa ≈ percentile: width should match a
        # plain percentile bootstrap of the mean within a generous tolerance.
        rng = np.random.default_rng(5)
        x = rng.normal(10.0, 3.0, size=80)
        lo, hi = bca_ci(x, rng=np.random.default_rng(5))
        # Reference percentile bootstrap with an independent stream.
        ref_rng = np.random.default_rng(5)
        boot = ref_rng.choice(x, size=(10_000, len(x)), replace=True).mean(axis=1)
        plo, phi = np.quantile(boot, [0.025, 0.975])
        assert (hi - lo) == pytest.approx(phi - plo, rel=0.20)

    def test_skewed_distribution_is_asymmetric(self):
        # The reason BCa exists: on skewed data the CI is not symmetric
        # about the point estimate.
        rng = np.random.default_rng(6)
        x = rng.lognormal(mean=0.0, sigma=1.0, size=60)
        m = float(np.mean(x))
        lo, hi = bca_ci(x, rng=np.random.default_rng(6))
        left, right = m - lo, hi - m
        assert abs(right - left) > 0.05 * (hi - lo)

    def test_n_boot_parameter_respected(self):
        x = [1.0, 2.0, 2.5, 3.0, 4.0, 5.5, 6.0]
        lo, hi = bca_ci(x, n_boot=500, rng=np.random.default_rng(8))
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
