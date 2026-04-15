"""Tests for metrics/statistical_inference.py — Phase 28.1."""

from __future__ import annotations

import numpy as np
import pytest

from metrics.statistical_inference import (
    apply_fdr_to_results,
    benjamini_hochberg,
    bootstrap_ci,
    report_metric,
)

# ── benjamini_hochberg ────────────────────────────────────────────────────────


class TestBenjaminiHochberg:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([0.01], alpha=1.5)

    def test_all_significant(self):
        p = [0.001, 0.002, 0.003]
        adj, rej = benjamini_hochberg(p)
        assert all(rej)

    def test_none_significant(self):
        p = [0.5, 0.6, 0.7, 0.8]
        adj, rej = benjamini_hochberg(p)
        assert not any(rej)

    def test_adjusted_geq_raw(self):
        """Adjusted p-values should be ≥ raw p-values (BH is conservative)."""
        p = [0.01, 0.04, 0.10, 0.30]
        adj, _ = benjamini_hochberg(p)
        for raw, a in zip(p, adj):
            assert a >= raw - 1e-10

    def test_adjusted_clipped_to_one(self):
        p = [0.99, 0.98, 0.97]
        adj, _ = benjamini_hochberg(p)
        assert all(v <= 1.0 for v in adj)

    def test_single_pvalue_significant(self):
        adj, rej = benjamini_hochberg([0.001])
        assert rej[0] is True

    def test_output_length_matches_input(self):
        p = [0.01, 0.05, 0.10]
        adj, rej = benjamini_hochberg(p)
        assert len(adj) == 3
        assert len(rej) == 3

    def test_monotone_adjusted(self):
        """BH procedure ensures non-decreasing adjusted p-values when sorted."""
        p = [0.005, 0.011, 0.022, 0.05, 0.10]
        adj, _ = benjamini_hochberg(p)
        # Sort by original rank and check adjusted values don't have inversions
        adj_arr = np.array(adj)
        # Each adjusted p should be ≥ the adjusted p of any smaller raw p
        sorted_indices = np.argsort(p)
        sorted_adj = adj_arr[sorted_indices]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-9

    def test_known_example(self):
        """Reproduce a known BH result: m=5 tests, rank-ordered p-values."""
        # BH on [0.001, 0.02, 0.04, 0.10, 0.50] at alpha=0.05
        # BH threshold for rank i: alpha * i / m = 0.05 * i / 5
        # rank 1: 0.01, rank 2: 0.02, rank 3: 0.03, rank 4: 0.04, rank 5: 0.05
        # 0.001 < 0.01 ✓, 0.02 ≤ 0.02 ✓, 0.04 > 0.03 ✗
        p = [0.001, 0.02, 0.04, 0.10, 0.50]
        adj, rej = benjamini_hochberg(p)
        assert rej[0] is True  # 0.001 significant
        assert rej[1] is True  # 0.02 significant


# ── bootstrap_ci ──────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_point_estimate_is_mean_by_default(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        point, lower, upper = bootstrap_ci(values, n_bootstrap=500)
        assert point == pytest.approx(np.mean(values))

    def test_ci_contains_point(self):
        values = list(range(50))
        point, lower, upper = bootstrap_ci(values, n_bootstrap=1000)
        assert lower <= point <= upper

    def test_wider_ci_for_higher_confidence(self):
        values = list(range(30))
        _, lo_90, hi_90 = bootstrap_ci(values, confidence=0.90, n_bootstrap=1000)
        _, lo_99, hi_99 = bootstrap_ci(values, confidence=0.99, n_bootstrap=1000)
        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99
        assert width_99 >= width_90

    def test_custom_stat_fn(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        point, _, _ = bootstrap_ci(values, stat_fn=np.median, n_bootstrap=500)
        assert point == pytest.approx(np.median(values))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], confidence=1.5)

    def test_reproducible_with_same_seed(self):
        values = list(range(50))
        r1 = bootstrap_ci(values, n_bootstrap=500, random_state=7)
        r2 = bootstrap_ci(values, n_bootstrap=500, random_state=7)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        values = list(range(50))
        _, lo1, hi1 = bootstrap_ci(values, n_bootstrap=200, random_state=1)
        _, lo2, hi2 = bootstrap_ci(values, n_bootstrap=200, random_state=99)
        # CIs should differ (with high probability for different seeds)
        # Allow they could be equal but practically they won't be
        assert not (lo1 == lo2 and hi1 == hi2)


# ── report_metric ─────────────────────────────────────────────────────────────


class TestReportMetric:
    def test_returns_required_keys(self):
        r = report_metric([0.6, 0.58, 0.64], n_bootstrap=500)
        for key in ["value", "lower", "upper", "ci_str", "n", "confidence"]:
            assert key in r

    def test_ci_str_format(self):
        r = report_metric([0.6, 0.6, 0.6], n_bootstrap=100)
        assert "[" in r["ci_str"] and "]" in r["ci_str"]

    def test_n_matches_input_length(self):
        r = report_metric([1.0, 2.0, 3.0], n_bootstrap=100)
        assert r["n"] == 3

    def test_lower_leq_value_leq_upper(self):
        r = report_metric(list(range(1, 21)), n_bootstrap=500)
        assert r["lower"] <= r["value"] <= r["upper"]

    def test_confidence_stored(self):
        r = report_metric([1.0, 2.0], confidence=0.90, n_bootstrap=100)
        assert r["confidence"] == 0.90


# ── apply_fdr_to_results ─────────────────────────────────────────────────────


class TestApplyFDR:
    def test_keys_preserved(self):
        named_p = {"h1_brm": 0.001, "h2_bias": 0.03, "h3_gini": 0.40}
        result = apply_fdr_to_results(named_p)
        assert set(result.keys()) == {"h1_brm", "h2_bias", "h3_gini"}

    def test_each_entry_has_three_keys(self):
        named_p = {"test_a": 0.01, "test_b": 0.50}
        result = apply_fdr_to_results(named_p)
        for v in result.values():
            assert "raw_p" in v and "adjusted_p" in v and "rejected" in v

    def test_significant_test_rejected(self):
        named_p = {"tiny_p": 0.0001, "large_p": 0.90}
        result = apply_fdr_to_results(named_p)
        assert result["tiny_p"]["rejected"] is True

    def test_large_p_not_rejected(self):
        named_p = {"large_p": 0.90}
        result = apply_fdr_to_results(named_p)
        assert result["large_p"]["rejected"] is False
