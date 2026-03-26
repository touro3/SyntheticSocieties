"""Tests for the Behavioral Realism Metric (BRM) and RLHF Bias Index.

Phase 23 — Formal Framework formalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics.behavioral_realism import (
    compute_brm_jsd,
    compute_composite_brm,
    compute_rlhf_bias_index,
    rlhf_bias_index_from_counts,
)


# ── BRM-JSD ──────────────────────────────────────────────────────────────


class TestBRMJSD:
    """BRM_JSD = 1 - JSD(D_sim || D_ESS).  Range [0, 1], 1 = perfect match."""

    def test_identical_distributions_returns_one(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        result = compute_brm_jsd(values, values)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_disjoint_distributions_low(self):
        sim = [0.0] * 100
        emp = [100.0] * 100
        result = compute_brm_jsd(sim, emp)
        # With histogram epsilon smoothing, JSD won't reach 1.0 for
        # point-mass distributions, so BRM won't reach 0.0 exactly.
        assert result < 0.5

    def test_bounded_zero_one(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            sim = rng.normal(0, 1, 200).tolist()
            emp = rng.normal(3, 2, 200).tolist()
            result = compute_brm_jsd(sim, emp)
            assert 0.0 <= result <= 1.0

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_brm_jsd([], [1.0, 2.0])
        with pytest.raises(ValueError):
            compute_brm_jsd([1.0], [])

    def test_single_value_each(self):
        # Degenerate but should not crash
        result = compute_brm_jsd([5.0], [5.0])
        assert 0.0 <= result <= 1.0

    def test_similar_distributions_higher_than_dissimilar(self):
        base = list(range(100))
        similar = [x + 1 for x in base]
        dissimilar = [x + 50 for x in base]
        brm_similar = compute_brm_jsd(base, similar)
        brm_dissimilar = compute_brm_jsd(base, dissimilar)
        assert brm_similar > brm_dissimilar


# ── RLHF Bias Index ─────────────────────────────────────────────────────


class TestRLHFBiasIndex:
    """B_RLHF = TV(pi, pi_uniform).  Range [0, 1], 0 = no bias."""

    def test_uniform_actions_zero_bias(self):
        dist = {"work": 1 / 3, "save": 1 / 3, "cooperate": 1 / 3}
        assert compute_rlhf_bias_index(dist) == pytest.approx(0.0, abs=1e-9)

    def test_all_cooperate_max_bias(self):
        dist = {"work": 0.0, "save": 0.0, "cooperate": 1.0}
        # TV = 0.5 * (|0-1/3| + |0-1/3| + |1-1/3|) = 0.5 * (1/3+1/3+2/3) = 2/3
        assert compute_rlhf_bias_index(dist) == pytest.approx(2 / 3, abs=1e-9)

    def test_all_work_max_bias(self):
        dist = {"work": 1.0, "save": 0.0, "cooperate": 0.0}
        assert compute_rlhf_bias_index(dist) == pytest.approx(2 / 3, abs=1e-9)

    def test_bounded_zero_one(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            raw = rng.dirichlet([1, 1, 1])
            dist = {"work": raw[0], "save": raw[1], "cooperate": raw[2]}
            result = compute_rlhf_bias_index(dist)
            assert 0.0 <= result <= 1.0

    def test_missing_action_treated_as_zero(self):
        dist = {"cooperate": 0.8, "work": 0.2}
        result = compute_rlhf_bias_index(dist)
        assert 0.0 <= result <= 1.0

    def test_empty_distribution_raises(self):
        with pytest.raises(ValueError):
            compute_rlhf_bias_index({})


class TestRLHFBiasIndexFromCounts:
    """Convenience wrapper accepting raw action counts."""

    def test_from_even_counts(self):
        counts = {"work": 10, "save": 10, "cooperate": 10}
        assert rlhf_bias_index_from_counts(counts) == pytest.approx(0.0, abs=1e-9)

    def test_from_skewed_counts(self):
        counts = {"work": 0, "save": 0, "cooperate": 100}
        assert rlhf_bias_index_from_counts(counts) == pytest.approx(2 / 3, abs=1e-9)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            rlhf_bias_index_from_counts({"work": 0, "save": 0, "cooperate": 0})


# ── Composite BRM ────────────────────────────────────────────────────────


class TestCompositeBRM:
    """Weighted aggregate of sub-metrics."""

    def test_perfect_match_returns_near_one(self):
        values = list(range(100))
        result = compute_composite_brm(
            sim_wealth=values,
            emp_wealth=values,
            sim_gini=0.35,
            emp_gini=0.35,
            sim_coop_rate=0.40,
            emp_coop_rate=0.40,
            temporal_stability_jsd=0.0,
        )
        assert result["composite"] == pytest.approx(1.0, abs=0.02)

    def test_returns_all_components(self):
        result = compute_composite_brm(
            sim_wealth=list(range(50)),
            emp_wealth=list(range(50, 100)),
            sim_gini=0.10,
            emp_gini=0.40,
            sim_coop_rate=0.80,
            emp_coop_rate=0.30,
            temporal_stability_jsd=0.3,
        )
        expected_keys = {
            "composite",
            "jsd_component",
            "gini_component",
            "coop_component",
            "stability_component",
        }
        assert expected_keys == set(result.keys())

    def test_all_components_bounded(self):
        result = compute_composite_brm(
            sim_wealth=[0.0] * 50,
            emp_wealth=[100.0] * 50,
            sim_gini=0.0,
            emp_gini=1.0,
            sim_coop_rate=0.0,
            emp_coop_rate=1.0,
            temporal_stability_jsd=1.0,
        )
        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of bounds"

    def test_custom_weights(self):
        weights = {"jsd": 0.5, "gini_gap": 0.2, "coop_gap": 0.2, "stability": 0.1}
        result = compute_composite_brm(
            sim_wealth=list(range(100)),
            emp_wealth=list(range(100)),
            sim_gini=0.35,
            emp_gini=0.35,
            sim_coop_rate=0.40,
            emp_coop_rate=0.40,
            temporal_stability_jsd=0.0,
            weights=weights,
        )
        assert result["composite"] == pytest.approx(1.0, abs=0.02)

    def test_invalid_weights_raises(self):
        bad_weights = {"jsd": 0.5, "gini_gap": 0.5, "coop_gap": 0.5, "stability": 0.5}
        with pytest.raises(ValueError, match="sum to 1"):
            compute_composite_brm(
                sim_wealth=[1.0],
                emp_wealth=[1.0],
                sim_gini=0.3,
                emp_gini=0.3,
                sim_coop_rate=0.3,
                emp_coop_rate=0.3,
                temporal_stability_jsd=0.0,
                weights=bad_weights,
            )

    def test_worse_match_lower_composite(self):
        good = compute_composite_brm(
            sim_wealth=list(range(100)),
            emp_wealth=list(range(100)),
            sim_gini=0.35,
            emp_gini=0.35,
            sim_coop_rate=0.40,
            emp_coop_rate=0.40,
            temporal_stability_jsd=0.01,
        )
        bad = compute_composite_brm(
            sim_wealth=[0.0] * 100,
            emp_wealth=[100.0] * 100,
            sim_gini=0.05,
            emp_gini=0.45,
            sim_coop_rate=0.90,
            emp_coop_rate=0.30,
            temporal_stability_jsd=0.5,
        )
        assert good["composite"] > bad["composite"]
