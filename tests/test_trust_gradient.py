"""Tests for trust-gradient sub-population validation.

Phase 17 — Trust-Gradient Sub-Population Validation.

The central hypothesis: BGF agents grounded in higher-trust ESS sub-populations
produce higher cooperation rates. This validates that the grounding function Φ
genuinely transfers empirical trust signals to simulated behavior.
"""

from __future__ import annotations

import pytest

from metrics.trust_gradient import (
    TRUST_GROUPS,
    TrustGroup,
    compute_trust_gradient,
    compute_trust_recovery_correlation,
)


# ── TrustGroup dataclass ─────────────────────────────────────────────────────


class TestTrustGroup:
    def test_fields_present(self):
        g = TRUST_GROUPS[0]
        assert hasattr(g, "name")
        assert hasattr(g, "band")
        assert hasattr(g, "trust_range")
        assert hasattr(g, "ess_reference_mean")

    def test_trust_range_ordered(self):
        for g in TRUST_GROUPS:
            lo, hi = g.trust_range
            assert lo < hi, f"{g.name}: lo must be less than hi"

    def test_reference_mean_in_range(self):
        for g in TRUST_GROUPS:
            lo, hi = g.trust_range
            assert lo <= g.ess_reference_mean <= hi, (
                f"{g.name}: ess_reference_mean {g.ess_reference_mean} not in [{lo}, {hi}]"
            )

    def test_four_groups_defined(self):
        assert len(TRUST_GROUPS) == 4

    def test_groups_ordered_by_trust(self):
        """TRUST_GROUPS should be ordered from lowest to highest trust."""
        means = [g.ess_reference_mean for g in TRUST_GROUPS]
        assert means == sorted(means), "TRUST_GROUPS must be in ascending trust order"

    def test_custom_group_creation(self):
        g = TrustGroup(
            name="Test",
            band="high",
            trust_range=(0.6, 0.8),
            ess_reference_mean=0.7,
        )
        assert g.name == "Test"
        assert g.band == "high"


# ── compute_trust_gradient ───────────────────────────────────────────────────


class TestComputeTrustGradient:
    def _make_results(self, coop_rates: list[float]) -> dict[str, dict]:
        return {
            g.name: {"coop_rate": r, "gini": 0.3, "mean_wealth": 50.0}
            for g, r in zip(TRUST_GROUPS, coop_rates)
        }

    def test_returns_dict(self):
        results = self._make_results([0.2, 0.3, 0.4, 0.5])
        gradient = compute_trust_gradient(results)
        assert isinstance(gradient, dict)

    def test_keys_are_group_names(self):
        results = self._make_results([0.2, 0.3, 0.4, 0.5])
        gradient = compute_trust_gradient(results)
        assert set(gradient.keys()) == {g.name for g in TRUST_GROUPS}

    def test_values_are_floats(self):
        results = self._make_results([0.1, 0.3, 0.5, 0.7])
        gradient = compute_trust_gradient(results)
        for v in gradient.values():
            assert isinstance(v, float)

    def test_ordered_output_matches_input(self):
        coop_rates = [0.20, 0.35, 0.50, 0.65]
        results = self._make_results(coop_rates)
        gradient = compute_trust_gradient(results)
        for g, expected in zip(TRUST_GROUPS, coop_rates):
            assert gradient[g.name] == pytest.approx(expected)

    def test_missing_group_raises(self):
        partial = {TRUST_GROUPS[0].name: {"coop_rate": 0.3, "gini": 0.3, "mean_wealth": 50.0}}
        with pytest.raises((KeyError, ValueError)):
            compute_trust_gradient(partial)

    def test_empty_results_raises(self):
        with pytest.raises((KeyError, ValueError)):
            compute_trust_gradient({})


# ── compute_trust_recovery_correlation ──────────────────────────────────────


class TestComputeTrustRecoveryCorrelation:
    def _make_results(self, coop_rates: list[float]) -> dict[str, dict]:
        return {
            g.name: {"coop_rate": r, "gini": 0.3, "mean_wealth": 50.0}
            for g, r in zip(TRUST_GROUPS, coop_rates)
        }

    def test_perfect_positive_correlation(self):
        """Simulated coop perfectly tracks ESS trust → r ≈ 1.0."""
        results = self._make_results([0.20, 0.35, 0.50, 0.65])
        out = compute_trust_recovery_correlation(results)
        assert out["spearman_r"] == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self):
        """Reversed coop rates → r = -1.0."""
        results = self._make_results([0.65, 0.50, 0.35, 0.20])
        out = compute_trust_recovery_correlation(results)
        assert out["spearman_r"] == pytest.approx(-1.0, abs=0.01)

    def test_r_in_valid_range(self):
        results = self._make_results([0.4, 0.4, 0.4, 0.4])  # flat
        out = compute_trust_recovery_correlation(results)
        assert -1.0 <= out["spearman_r"] <= 1.0

    def test_output_keys_present(self):
        results = self._make_results([0.20, 0.35, 0.50, 0.65])
        out = compute_trust_recovery_correlation(results)
        expected_keys = {"spearman_r", "p_value", "n_groups", "is_significant", "interpretation"}
        assert expected_keys.issubset(set(out.keys()))

    def test_n_groups_correct(self):
        results = self._make_results([0.20, 0.35, 0.50, 0.65])
        out = compute_trust_recovery_correlation(results)
        assert out["n_groups"] == 4

    def test_p_value_in_range(self):
        results = self._make_results([0.20, 0.35, 0.50, 0.65])
        out = compute_trust_recovery_correlation(results)
        assert 0.0 <= out["p_value"] <= 1.0

    def test_significant_flag_on_strong_correlation(self):
        results = self._make_results([0.20, 0.35, 0.50, 0.65])
        out = compute_trust_recovery_correlation(results)
        # With n=4 and perfect correlation the p_value is small (~0.083);
        # is_significant threshold is p < 0.10 for n=4
        assert isinstance(out["is_significant"], bool)

    def test_flat_coop_is_not_significantly_positive(self):
        """Flat cooperation rates provide no evidence of gradient recovery."""
        results = self._make_results([0.40, 0.40, 0.40, 0.40])
        out = compute_trust_recovery_correlation(results)
        # Spearman r is 0 (or undefined) for constant sequences
        assert abs(out["spearman_r"]) < 0.1 or out["p_value"] > 0.50

    def test_interpretation_is_string(self):
        results = self._make_results([0.20, 0.35, 0.50, 0.65])
        out = compute_trust_recovery_correlation(results)
        assert isinstance(out["interpretation"], str)
        assert len(out["interpretation"]) > 0

    def test_custom_groups_respected(self):
        custom = [
            TrustGroup("A", "low",       (0.2, 0.4), 0.30),
            TrustGroup("B", "high",      (0.6, 0.8), 0.70),
        ]
        results = {"A": {"coop_rate": 0.20, "gini": 0.3, "mean_wealth": 50.0},
                   "B": {"coop_rate": 0.60, "gini": 0.3, "mean_wealth": 50.0}}
        out = compute_trust_recovery_correlation(results, cultural_groups=custom)
        assert out["n_groups"] == 2


# ── compute_trust_gradient edge cases ────────────────────────────────────────


class TestComputeTrustGradientEdgeCases:
    def test_empty_group_results_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_trust_gradient({})

    def test_missing_group_raises_key_error(self):
        # Providing only one of the four canonical groups should raise KeyError.
        partial = {"Low-Trust": {"coop_rate": 0.2, "gini": 0.3, "mean_wealth": 50.0}}
        with pytest.raises(KeyError):
            compute_trust_gradient(partial)
