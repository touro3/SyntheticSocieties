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
    ClusterSimResult,
    CrossCulturalResult,
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
            compute_cross_cultural_correlation(
                [ClusterSimResult("x", 0.5, 0.3, 0.3, 10, 5)]
            )

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
