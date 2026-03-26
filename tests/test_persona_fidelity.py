"""Tests for metrics/persona_fidelity.py — PCA, composite scores, fidelity reports."""

import json

import numpy as np
import pandas as pd
import pytest

from metrics.persona_fidelity import (
    composite_score,
    compute_fidelity_report,
    compute_pca_projection,
    summarize_synthetic_runs,
    write_report_files,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

TARGET_ITEMS = [
    {"name": "trust_people"},
    {"name": "risk_tolerance"},
]

TARGET_ITEMS_WITH_INVERSE = [
    {"name": "trust_people"},
    {"name": "competitiveness", "inverse": True},
]


def _make_real_profiles(n: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "profile_id": [f"p{i}" for i in range(n)],
        "real_trust_people": rng.uniform(0.1, 0.9, n),
        "real_risk_tolerance": rng.uniform(0.1, 0.9, n),
        "real_competitiveness": rng.uniform(0.1, 0.9, n),
    })


def _make_synth_profiles(real: pd.DataFrame, noise: float = 0.05, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    synth = pd.DataFrame({"profile_id": real["profile_id"]})
    for col in real.columns:
        if col == "profile_id":
            continue
        base_col = col.replace("real_", "")
        synth[base_col] = np.clip(real[col].values + rng.normal(0, noise, len(real)), 0, 1)
    return synth


# ── composite_score ──────────────────────────────────────────────────────────


class TestCompositeScore:
    def test_basic_mean(self):
        df = pd.DataFrame({"trust_people": [0.5, 1.0], "risk_tolerance": [0.5, 0.0]})
        scores = composite_score(df, TARGET_ITEMS)
        expected = np.array([(0.5 + 0.5) / 2, (1.0 + 0.0) / 2]) * 100.0
        np.testing.assert_allclose(scores.values, expected)

    def test_inverse_flag(self):
        df = pd.DataFrame({"trust_people": [0.8], "competitiveness": [0.2]})
        scores = composite_score(df, TARGET_ITEMS_WITH_INVERSE)
        # competitiveness inverted: 1 - 0.2 = 0.8. Mean = (0.8 + 0.8)/2 = 0.8 * 100 = 80
        assert scores.iloc[0] == pytest.approx(80.0)

    def test_clips_values(self):
        df = pd.DataFrame({"trust_people": [1.5], "risk_tolerance": [-0.3]})
        scores = composite_score(df, TARGET_ITEMS)
        # Clipped to [0,1]: (1.0 + 0.0)/2 * 100 = 50
        assert scores.iloc[0] == pytest.approx(50.0)

    def test_prefix(self):
        df = pd.DataFrame({"real_trust_people": [0.6], "real_risk_tolerance": [0.4]})
        scores = composite_score(df, TARGET_ITEMS, prefix="real_")
        assert scores.iloc[0] == pytest.approx(50.0)

    def test_all_zeros(self):
        df = pd.DataFrame({"trust_people": [0.0], "risk_tolerance": [0.0]})
        scores = composite_score(df, TARGET_ITEMS)
        assert scores.iloc[0] == pytest.approx(0.0)

    def test_all_ones(self):
        df = pd.DataFrame({"trust_people": [1.0], "risk_tolerance": [1.0]})
        scores = composite_score(df, TARGET_ITEMS)
        assert scores.iloc[0] == pytest.approx(100.0)


# ── compute_pca_projection ──────────────────────────────────────────────────


class TestPCAProjection:
    def test_returns_expected_keys(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        result = compute_pca_projection(real, synth, TARGET_ITEMS)
        for key in ["explained_variance_pc1_proxy", "real_pc1", "synthetic_pc1",
                     "pc1_pearson", "pc1_spearman", "pc1_bias"]:
            assert key in result

    def test_identical_data_has_near_perfect_correlation(self):
        real = _make_real_profiles()
        # Synthetic == real (zero noise)
        synth = _make_synth_profiles(real, noise=0.0)
        result = compute_pca_projection(real, synth, TARGET_ITEMS)
        assert result["pc1_pearson"] > 0.99
        assert abs(result["pc1_bias"]) < 0.01

    def test_pc1_lengths_match_data(self):
        real = _make_real_profiles(n=15)
        synth = _make_synth_profiles(real)
        result = compute_pca_projection(real, synth, TARGET_ITEMS)
        assert len(result["real_pc1"]) == 15
        assert len(result["synthetic_pc1"]) == 15

    def test_explained_variance_is_bounded(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        result = compute_pca_projection(real, synth, TARGET_ITEMS)
        assert 0.0 <= result["explained_variance_pc1_proxy"] <= 1.0

    def test_constant_column_handled(self):
        """If a real column has zero std, sigma is replaced with 1.0."""
        real = _make_real_profiles()
        real["real_trust_people"] = 0.5  # constant
        synth = _make_synth_profiles(real, noise=0.01)
        # Should not raise / produce NaN
        result = compute_pca_projection(real, synth, TARGET_ITEMS)
        assert np.isfinite(result["pc1_pearson"])

    def test_high_noise_lowers_correlation(self):
        real = _make_real_profiles(n=50, seed=0)
        synth_low = _make_synth_profiles(real, noise=0.01)
        synth_high = _make_synth_profiles(real, noise=0.5)
        r_low = compute_pca_projection(real, synth_low, TARGET_ITEMS)
        r_high = compute_pca_projection(real, synth_high, TARGET_ITEMS)
        assert r_low["pc1_pearson"] > r_high["pc1_pearson"]


# ── summarize_synthetic_runs ─────────────────────────────────────────────────


class TestSummarizeSyntheticRuns:
    def test_aggregation(self):
        df = pd.DataFrame({
            "profile_id": ["p0", "p0", "p1"],
            "replication_seed": [1, 2, 1],
            "trust_people": [0.4, 0.6, 0.8],
            "risk_tolerance": [0.3, 0.5, 0.7],
        })
        result = summarize_synthetic_runs(df, TARGET_ITEMS)
        assert len(result) == 2
        assert "synthetic_score_0_100" in result.columns
        # p0 trust_people mean = 0.5, risk_tolerance mean = 0.4 → score = 45
        p0 = result[result.profile_id == "p0"].iloc[0]
        assert p0["n_synthetic"] == 2
        assert p0["synthetic_score_0_100"] == pytest.approx(45.0)


# ── compute_fidelity_report ──────────────────────────────────────────────────


class TestComputeFidelityReport:
    def test_returns_report_and_dataframe(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        report, df = compute_fidelity_report(real, synth, TARGET_ITEMS)
        assert isinstance(report, dict)
        assert isinstance(df, pd.DataFrame)

    def test_report_structure(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        report, _ = compute_fidelity_report(real, synth, TARGET_ITEMS)
        assert "n_profiles" in report
        assert "score_metrics" in report
        assert "item_metrics" in report
        assert "pca_metrics" in report
        assert "affine_recalibration" in report
        assert "variance_scaling" in report

    def test_score_metrics_keys(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        report, _ = compute_fidelity_report(real, synth, TARGET_ITEMS)
        sm = report["score_metrics"]
        for key in ["mean_real_score", "mean_synthetic_score", "score_bias",
                     "score_mae", "score_rmse", "dispersion_ratio", "pearson", "spearman"]:
            assert key in sm

    def test_item_metrics_per_target(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        report, _ = compute_fidelity_report(real, synth, TARGET_ITEMS)
        for item in TARGET_ITEMS:
            assert item["name"] in report["item_metrics"]
            im = report["item_metrics"][item["name"]]
            for key in ["mae", "bias", "pearson", "spearman"]:
                assert key in im

    def test_affine_reduces_bias(self):
        """Affine recalibration should reduce bias toward zero."""
        real = _make_real_profiles(n=50)
        synth = _make_synth_profiles(real, noise=0.1)
        report, _ = compute_fidelity_report(real, synth, TARGET_ITEMS)
        raw_bias = abs(report["score_metrics"]["score_bias"])
        affine_bias = abs(report["affine_recalibration"]["bias_after"])
        assert affine_bias <= raw_bias + 0.01  # affine should not worsen bias

    def test_n_profiles_matches(self):
        real = _make_real_profiles(n=8)
        synth = _make_synth_profiles(real)
        report, df = compute_fidelity_report(real, synth, TARGET_ITEMS)
        assert report["n_profiles"] == 8
        assert len(df) == 8

    def test_per_profile_df_has_diff_columns(self):
        real = _make_real_profiles()
        synth = _make_synth_profiles(real)
        _, df = compute_fidelity_report(real, synth, TARGET_ITEMS)
        assert "score_diff" in df.columns
        assert "abs_score_diff" in df.columns
        for item in TARGET_ITEMS:
            assert f"diff_{item['name']}" in df.columns


# ── write_report_files ───────────────────────────────────────────────────────


class TestWriteReportFiles:
    def test_creates_files(self, tmp_path):
        real = _make_real_profiles(n=5)
        synth = _make_synth_profiles(real)
        report, df = compute_fidelity_report(real, synth, TARGET_ITEMS)
        write_report_files(tmp_path / "run1", report, df, synth)

        assert (tmp_path / "run1" / "fidelity_report.json").exists()
        assert (tmp_path / "run1" / "per_profile_comparison.csv").exists()
        assert (tmp_path / "run1" / "synthetic_profile_summary.csv").exists()

    def test_json_is_valid(self, tmp_path):
        real = _make_real_profiles(n=5)
        synth = _make_synth_profiles(real)
        report, df = compute_fidelity_report(real, synth, TARGET_ITEMS)
        write_report_files(tmp_path / "run2", report, df, synth)

        loaded = json.loads((tmp_path / "run2" / "fidelity_report.json").read_text())
        assert loaded["n_profiles"] == 5
