"""Tests for statistical significance functions in tracker/analytics.py."""

import numpy as np
import pandas as pd
import pytest

from tracker.analytics import (
    bootstrap_ci,
    cohens_d,
    mann_whitney_test,
    pairwise_significance,
)


# ── Cohen's d ────────────────────────────────────────────────────────────────


class TestCohensD:
    def test_identical_groups_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_known_large_effect(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(a, b)
        # Mean diff = 9, pooled std ≈ 1.58 → d ≈ 5.7 (very large)
        assert d > 5.0

    def test_sign_reflects_direction(self):
        a = np.array([9.0, 10.0, 11.0])
        b = np.array([19.0, 20.0, 21.0])
        assert cohens_d(a, b) < 0  # a < b
        assert cohens_d(b, a) > 0  # b > a

    def test_small_sample_returns_zero(self):
        assert cohens_d(np.array([1.0]), np.array([2.0])) == 0.0

    def test_zero_variance_returns_zero(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cohens_d(a, b) == 0.0

    def test_conventional_medium_effect(self):
        """Cohen's d ≈ 0.5 is considered a medium effect."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0.5, 1, 1000)
        d = cohens_d(a, b)
        assert -0.7 < d < -0.3  # a.mean < b.mean → negative d


# ── Mann-Whitney U ───────────────────────────────────────────────────────────


class TestMannWhitneyTest:
    def test_identical_groups_high_p(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mann_whitney_test(a, a)
        assert result["p_value"] > 0.05

    def test_separated_groups_low_p(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        result = mann_whitney_test(a, b)
        assert result["p_value"] < 0.05

    def test_returns_expected_keys(self):
        result = mann_whitney_test(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert "U_statistic" in result
        assert "p_value" in result

    def test_empty_group_returns_p_one(self):
        result = mann_whitney_test(np.array([]), np.array([1.0]))
        assert result["p_value"] == 1.0

    def test_single_values_each(self):
        result = mann_whitney_test(np.array([1.0]), np.array([100.0]))
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_returns_expected_keys(self):
        result = bootstrap_ci(np.array([1.0, 2.0, 3.0]))
        for key in ["estimate", "ci_lower", "ci_upper", "confidence"]:
            assert key in result

    def test_ci_contains_estimate(self):
        result = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]

    def test_empty_data(self):
        result = bootstrap_ci(np.array([]))
        assert result["estimate"] == 0.0
        assert result["ci_lower"] == 0.0

    def test_single_value(self):
        result = bootstrap_ci(np.array([42.0]))
        assert result["estimate"] == 42.0
        assert result["ci_lower"] == 42.0
        assert result["ci_upper"] == 42.0

    def test_wider_ci_with_more_variance(self):
        tight = bootstrap_ci(np.array([10.0, 10.0, 10.0, 10.0, 10.0]), seed=0)
        wide = bootstrap_ci(np.array([1.0, 5.0, 10.0, 50.0, 100.0]), seed=0)
        tight_width = tight["ci_upper"] - tight["ci_lower"]
        wide_width = wide["ci_upper"] - wide["ci_lower"]
        assert wide_width > tight_width

    def test_confidence_level_stored(self):
        result = bootstrap_ci(np.array([1.0, 2.0, 3.0]), confidence=0.99)
        assert result["confidence"] == 0.99

    def test_deterministic_with_seed(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = bootstrap_ci(data, seed=123)
        r2 = bootstrap_ci(data, seed=123)
        assert r1 == r2


# ── Pairwise Significance ───────────────────────────────────────────────────


class TestPairwiseSignificance:
    def test_basic_structure(self):
        df = pd.DataFrame({
            "policy_type": ["llm"] * 5 + ["random"] * 5,
            "wealth_mean": [100, 110, 105, 108, 112, 50, 55, 45, 52, 48],
        })
        result = pairwise_significance(df)
        assert len(result) == 1
        assert result.iloc[0]["group"] == "random"
        assert "cohens_d" in result.columns
        assert "p_value" in result.columns

    def test_significant_difference(self):
        df = pd.DataFrame({
            "policy_type": ["llm"] * 10 + ["random"] * 10,
            "wealth_mean": [100] * 10 + [10] * 10,
        })
        result = pairwise_significance(df)
        assert result.iloc[0]["significant_005"] == True

    def test_no_significant_difference(self):
        # Interleave identical values so there's zero difference
        vals = [100.0, 105.0, 95.0, 110.0, 90.0]
        df = pd.DataFrame({
            "policy_type": ["llm"] * 5 + ["random"] * 5,
            "wealth_mean": vals + vals,
        })
        result = pairwise_significance(df)
        assert result.iloc[0]["significant_005"] == False

    def test_multiple_groups(self):
        df = pd.DataFrame({
            "policy_type": ["llm"] * 5 + ["random"] * 5 + ["template"] * 5,
            "wealth_mean": [100] * 5 + [50] * 5 + [75] * 5,
        })
        result = pairwise_significance(df)
        assert len(result) == 2
        groups = set(result["group"].tolist())
        assert groups == {"random", "template"}

    def test_missing_reference_returns_empty(self):
        df = pd.DataFrame({
            "policy_type": ["random"] * 5,
            "wealth_mean": [50] * 5,
        })
        result = pairwise_significance(df, reference_group="llm")
        assert len(result) == 0

    def test_custom_metric_col(self):
        df = pd.DataFrame({
            "policy_type": ["llm"] * 5 + ["random"] * 5,
            "wealth_mean": [100] * 5 + [50] * 5,
            "stress_mean": [0.5] * 5 + [0.9] * 5,
        })
        result = pairwise_significance(df, metric_col="stress_mean")
        assert len(result) == 1
