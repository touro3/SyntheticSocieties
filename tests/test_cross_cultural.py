"""Tests for Phase 27 — True Cross-Cultural ESS Validation.

Tests cover:
- CountryCluster dataclass and load_clusters()
- ClusterSimResult and CrossCulturalResult dataclasses
- compute_cross_cultural_correlation() with gradient, inverse, and flat cases
- format_cross_cultural_table()
- cross_cultural_benchmarks.json integrity
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metrics.cross_cultural import (
    ClusterMultiSeedResult,
    ClusterSimResult,
    CrossCulturalResult,
    compare_holdout_fit,
    compute_benchmark_fit,
    compute_cross_cultural_correlation,
    format_cross_cultural_table,
)
from population.country_clusters import (
    CANONICAL_CLUSTER_ORDER,
    CountryCluster,
    get_cluster_by_name,
    load_clusters,
)

BENCHMARKS_PATH = Path("data/cross_cultural_benchmarks.json")


# ── benchmarks.json integrity ─────────────────────────────────────────────────


class TestBenchmarksJSON:
    def test_file_exists(self):
        assert BENCHMARKS_PATH.exists(), "cross_cultural_benchmarks.json missing"

    def test_has_three_clusters(self):
        data = json.loads(BENCHMARKS_PATH.read_text())
        assert len(data["clusters"]) == 3

    def test_cluster_names(self):
        data = json.loads(BENCHMARKS_PATH.read_text())
        assert set(data["clusters"].keys()) == {"nordic", "southern", "eastern"}

    def test_trust_values_in_range(self):
        data = json.loads(BENCHMARKS_PATH.read_text())
        for name, c in data["clusters"].items():
            v = c["ess_mean_trust_people"]
            assert 0.0 < v < 1.0, f"{name}: trust {v} out of (0, 1)"

    def test_nordic_highest_trust(self):
        data = json.loads(BENCHMARKS_PATH.read_text())
        c = data["clusters"]
        assert c["nordic"]["ess_mean_trust_people"] > c["southern"]["ess_mean_trust_people"]
        assert c["southern"]["ess_mean_trust_people"] > c["eastern"]["ess_mean_trust_people"]

    def test_trust_bands_correct(self):
        data = json.loads(BENCHMARKS_PATH.read_text())
        assert data["clusters"]["nordic"]["trust_band"] == "high"
        assert data["clusters"]["southern"]["trust_band"] == "moderate"
        assert data["clusters"]["eastern"]["trust_band"] == "low"

    def test_has_source_field(self):
        data = json.loads(BENCHMARKS_PATH.read_text())
        assert "source" in data


# ── CountryCluster ─────────────────────────────────────────────────────────────


class TestCountryCluster:
    def test_fields_present(self):
        c = CountryCluster(
            name="nordic",
            countries=["NO", "SE"],
            ess_mean_trust=0.67,
            ess_sd_trust=0.19,
            trust_band="high",
            description="test",
        )
        assert c.name == "nordic"
        assert c.ess_mean_trust == 0.67
        assert c.trust_band == "high"

    def test_load_clusters_returns_three(self):
        clusters = load_clusters()
        assert len(clusters) == 3

    def test_load_clusters_names(self):
        clusters = load_clusters()
        names = {c.name for c in clusters}
        assert names == {"nordic", "southern", "eastern"}

    def test_load_clusters_trust_ordered(self):
        clusters = load_clusters()
        by_name = {c.name: c for c in clusters}
        assert by_name["nordic"].ess_mean_trust > by_name["southern"].ess_mean_trust
        assert by_name["southern"].ess_mean_trust > by_name["eastern"].ess_mean_trust

    def test_canonical_order_ascending(self):
        assert CANONICAL_CLUSTER_ORDER == ["eastern", "southern", "nordic"]

    def test_get_cluster_by_name(self):
        c = get_cluster_by_name("nordic")
        assert c.name == "nordic"
        assert c.trust_band == "high"

    def test_get_cluster_unknown_raises(self):
        with pytest.raises(KeyError):
            get_cluster_by_name("unknown_cluster")


# ── CrossCultural metrics ─────────────────────────────────────────────────────


@pytest.fixture
def gradient_results() -> list[ClusterSimResult]:
    """Perfect positive gradient: trust rank == cooperation rank."""
    return [
        ClusterSimResult(
            cluster_name="nordic",
            ess_mean_trust=0.673,
            simulated_cooperation_rate=0.55,
            simulated_gini=0.28,
            n_agents=20,
            n_rounds=10,
        ),
        ClusterSimResult(
            cluster_name="southern",
            ess_mean_trust=0.463,
            simulated_cooperation_rate=0.40,
            simulated_gini=0.32,
            n_agents=20,
            n_rounds=10,
        ),
        ClusterSimResult(
            cluster_name="eastern",
            ess_mean_trust=0.421,
            simulated_cooperation_rate=0.30,
            simulated_gini=0.35,
            n_agents=20,
            n_rounds=10,
        ),
    ]


@pytest.fixture
def flat_results() -> list[ClusterSimResult]:
    """No gradient: all cooperation rates identical."""
    return [
        ClusterSimResult("nordic", 0.673, 0.40, 0.30, 20, 10),
        ClusterSimResult("southern", 0.463, 0.40, 0.30, 20, 10),
        ClusterSimResult("eastern", 0.421, 0.40, 0.30, 20, 10),
    ]


@pytest.fixture
def inverse_results() -> list[ClusterSimResult]:
    """Inverse gradient: higher trust → lower cooperation (grounding failure)."""
    return [
        ClusterSimResult("nordic", 0.673, 0.25, 0.38, 20, 10),
        ClusterSimResult("southern", 0.463, 0.40, 0.32, 20, 10),
        ClusterSimResult("eastern", 0.421, 0.55, 0.28, 20, 10),
    ]


class TestComputeCorrelation:
    def test_gradient_spearman_positive(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        assert result.spearman_rho > 0

    def test_gradient_pearson_positive(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        assert result.pearson_r > 0

    def test_flat_not_recovered(self, flat_results):
        result = compute_cross_cultural_correlation(flat_results)
        assert not result.gradient_recovered

    def test_inverse_spearman_negative(self, inverse_results):
        result = compute_cross_cultural_correlation(inverse_results)
        assert result.spearman_rho < 0

    def test_inverse_not_recovered(self, inverse_results):
        result = compute_cross_cultural_correlation(inverse_results)
        assert not result.gradient_recovered

    def test_returns_correct_type(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        assert isinstance(result, CrossCulturalResult)

    def test_requires_minimum_three_clusters(self):
        with pytest.raises(ValueError, match="at least 3"):
            compute_cross_cultural_correlation([ClusterSimResult("x", 0.5, 0.3, 0.3, 10, 5)])

    def test_requires_minimum_two_clusters(self):
        with pytest.raises(ValueError):
            compute_cross_cultural_correlation(
                [
                    ClusterSimResult("x", 0.5, 0.3, 0.3, 10, 5),
                    ClusterSimResult("y", 0.6, 0.4, 0.3, 10, 5),
                ]
            )

    def test_gradient_recovered_flag_on_perfect_gradient(self, gradient_results):
        # Perfect rank correlation (rho=1.0) gives p=0 for n=3
        result = compute_cross_cultural_correlation(gradient_results)
        assert result.gradient_recovered

    def test_p_values_in_range(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        assert 0.0 <= result.pearson_p <= 1.0
        assert 0.0 <= result.spearman_p <= 1.0

    def test_cluster_results_preserved(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        assert len(result.cluster_results) == 3


class TestFormatTable:
    def test_output_is_string(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        table = format_cross_cultural_table(result)
        assert isinstance(table, str)

    def test_contains_pearson(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        table = format_cross_cultural_table(result)
        assert "Pearson" in table

    def test_contains_spearman(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        table = format_cross_cultural_table(result)
        assert "Spearman" in table

    def test_contains_cluster_names(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        table = format_cross_cultural_table(result)
        assert "nordic" in table
        assert "southern" in table
        assert "eastern" in table

    def test_contains_gradient_verdict(self, gradient_results):
        result = compute_cross_cultural_correlation(gradient_results)
        table = format_cross_cultural_table(result)
        assert "Gradient recovered" in table


@pytest.fixture
def grounded_multiseed() -> list[ClusterMultiSeedResult]:
    return [
        ClusterMultiSeedResult(
            "eastern",
            0.418,
            0.280,
            0.02,
            0.260,
            0.300,
            3,
            [0.27, 0.29, 0.28],
            mean_gini=0.34,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=24.0,
        ),
        ClusterMultiSeedResult(
            "southern",
            0.455,
            0.350,
            0.02,
            0.330,
            0.370,
            3,
            [0.34, 0.36, 0.35],
            mean_gini=0.32,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=29.0,
        ),
        ClusterMultiSeedResult(
            "western",
            0.504,
            0.420,
            0.02,
            0.400,
            0.440,
            3,
            [0.41, 0.43, 0.42],
            mean_gini=0.30,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=37.0,
        ),
        ClusterMultiSeedResult(
            "anglo",
            0.565,
            0.500,
            0.02,
            0.480,
            0.520,
            3,
            [0.49, 0.51, 0.50],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=43.0,
        ),
        ClusterMultiSeedResult(
            "northern",
            0.634,
            0.580,
            0.02,
            0.560,
            0.600,
            3,
            [0.57, 0.59, 0.58],
            mean_gini=0.28,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=55.0,
        ),
        ClusterMultiSeedResult(
            "nordic",
            0.689,
            0.660,
            0.02,
            0.640,
            0.680,
            3,
            [0.65, 0.67, 0.66],
            mean_gini=0.27,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=68.0,
        ),
    ]


@pytest.fixture
def control_multiseed() -> list[ClusterMultiSeedResult]:
    return [
        ClusterMultiSeedResult(
            "eastern",
            0.418,
            0.460,
            0.01,
            0.450,
            0.470,
            3,
            [0.46, 0.47, 0.45],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=24.0,
        ),
        ClusterMultiSeedResult(
            "southern",
            0.455,
            0.450,
            0.01,
            0.440,
            0.460,
            3,
            [0.45, 0.46, 0.44],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=29.0,
        ),
        ClusterMultiSeedResult(
            "western",
            0.504,
            0.455,
            0.01,
            0.445,
            0.465,
            3,
            [0.45, 0.46, 0.46],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=37.0,
        ),
        ClusterMultiSeedResult(
            "anglo",
            0.565,
            0.448,
            0.01,
            0.438,
            0.458,
            3,
            [0.44, 0.45, 0.45],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=43.0,
        ),
        ClusterMultiSeedResult(
            "northern",
            0.634,
            0.452,
            0.01,
            0.442,
            0.462,
            3,
            [0.45, 0.46, 0.44],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=55.0,
        ),
        ClusterMultiSeedResult(
            "nordic",
            0.689,
            0.449,
            0.01,
            0.439,
            0.459,
            3,
            [0.45, 0.44, 0.46],
            mean_gini=0.29,
            n_agents=20,
            n_rounds=10,
            wvs_trust_pct=68.0,
        ),
    ]


class TestHoldoutFit:
    def test_ess_fit_positive_gradient(self, grounded_multiseed):
        fit = compute_benchmark_fit(grounded_multiseed, benchmark="ess")
        assert fit.pearson_r > 0
        assert fit.spearman_rho > 0

    def test_wvs_fit_positive_gradient(self, grounded_multiseed):
        fit = compute_benchmark_fit(grounded_multiseed, benchmark="wvs")
        assert fit.pearson_r > 0
        assert fit.spearman_rho > 0
        assert fit.rmse < 0.30

    def test_compare_holdout_prefers_grounded(self, grounded_multiseed, control_multiseed):
        cmp = compare_holdout_fit(grounded_multiseed, control_multiseed, benchmark="wvs")
        assert cmp.delta_pearson_r > 0
        assert cmp.delta_spearman_rho > 0
        assert cmp.delta_rmse > 0
        assert cmp.grounded_better

    def test_wvs_requires_wvs_values(self, grounded_multiseed):
        broken = list(grounded_multiseed)
        broken[0].wvs_trust_pct = None
        with pytest.raises(ValueError, match="missing wvs_trust_pct"):
            compute_benchmark_fit(broken, benchmark="wvs")
