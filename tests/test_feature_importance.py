"""Tests for metrics/feature_importance.py — Phase 28.3."""

from __future__ import annotations

import numpy as np
import pytest

from metrics.feature_importance import (
    ESS_FEATURE_NAMES,
    PROFILE_FULL,
    PROFILE_MEDIUM,
    PROFILE_MINIMAL,
    AgentRoundRecord,
    FeatureImportanceResult,
    build_feature_matrix,
    compute_ablation_table,
    run_logistic_regression,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_record(cooperated: int = 0, **overrides) -> AgentRoundRecord:
    defaults = {
        "trust_people": 0.5,
        "trust_institutions": 0.5,
        "risk_tolerance": 0.5,
        "social_activity": 0.5,
        "life_satisfaction": 0.5,
        "happiness": 0.5,
        "competitiveness": 0.5,
        "leadership_preference": 0.5,
        "health_status": 0.5,
        "religiosity": 0.5,
        "political_orientation": 0.5,
        "immigration_attitude": 0.5,
        "cooperated": cooperated,
    }
    defaults.update(overrides)
    return AgentRoundRecord(**defaults)


def _make_records_with_signal(n: int = 100) -> list[AgentRoundRecord]:
    """High trust → cooperate, low trust → not cooperate (clear signal)."""
    rng = np.random.default_rng(42)
    records = []
    for _ in range(n // 2):
        records.append(_make_record(cooperated=1, trust_people=rng.uniform(0.7, 1.0)))
        records.append(_make_record(cooperated=0, trust_people=rng.uniform(0.0, 0.3)))
    return records


# ── ESS_FEATURE_NAMES ─────────────────────────────────────────────────────────


class TestFeatureNames:
    def test_contains_trust_people(self):
        assert "trust_people" in ESS_FEATURE_NAMES

    def test_contains_risk_tolerance(self):
        assert "risk_tolerance" in ESS_FEATURE_NAMES

    def test_twelve_features(self):
        assert len(ESS_FEATURE_NAMES) == 12

    def test_no_duplicates(self):
        assert len(set(ESS_FEATURE_NAMES)) == len(ESS_FEATURE_NAMES)

    def test_profile_levels_are_subsets(self):
        for lvl in [PROFILE_MINIMAL, PROFILE_MEDIUM, PROFILE_FULL]:
            assert set(lvl).issubset(set(ESS_FEATURE_NAMES))

    def test_profile_depth_ordering(self):
        assert len(PROFILE_MINIMAL) < len(PROFILE_MEDIUM) < len(PROFILE_FULL)


# ── AgentRoundRecord ──────────────────────────────────────────────────────────


class TestAgentRoundRecord:
    def test_default_construction(self):
        rec = _make_record()
        assert rec.cooperated == 0

    def test_cooperated_flag(self):
        rec = _make_record(cooperated=1)
        assert rec.cooperated == 1

    def test_fields_match_feature_names(self):
        rec = _make_record()
        for feat in ESS_FEATURE_NAMES:
            assert hasattr(rec, feat), f"Missing field: {feat}"


# ── build_feature_matrix ──────────────────────────────────────────────────────


class TestBuildFeatureMatrix:
    def test_shape(self):
        records = [_make_record(cooperated=i % 2) for i in range(20)]
        X, y = build_feature_matrix(records)
        assert X.shape == (20, len(ESS_FEATURE_NAMES))
        assert y.shape == (20,)

    def test_subset_features(self):
        records = [_make_record() for _ in range(10)]
        X, y = build_feature_matrix(records, feature_names=["trust_people", "risk_tolerance"])
        assert X.shape == (10, 2)

    def test_y_binary(self):
        records = [_make_record(cooperated=i % 2) for i in range(10)]
        _, y = build_feature_matrix(records)
        assert set(y.tolist()).issubset({0, 1})

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="No records"):
            build_feature_matrix([])

    def test_unknown_feature_raises(self):
        records = [_make_record()]
        with pytest.raises(ValueError, match="Unknown feature"):
            build_feature_matrix(records, feature_names=["not_a_real_feature"])

    def test_trust_column_matches(self):
        records = [_make_record(trust_people=0.99)]
        X, _ = build_feature_matrix(records, feature_names=["trust_people"])
        assert X[0, 0] == pytest.approx(0.99)


# ── run_logistic_regression ───────────────────────────────────────────────────


class TestRunLogisticRegression:
    def test_returns_result(self):
        records = _make_records_with_signal(100)
        X, y = build_feature_matrix(records, feature_names=["trust_people"])
        result = run_logistic_regression(X, y, feature_names=["trust_people"])
        assert isinstance(result, FeatureImportanceResult)

    def test_trust_positive_coefficient(self):
        """High trust → cooperate implies positive coefficient for trust."""
        records = _make_records_with_signal(200)
        X, y = build_feature_matrix(records, feature_names=["trust_people"])
        result = run_logistic_regression(X, y, feature_names=["trust_people"])
        coef = result.coefficients[0]
        assert coef.coefficient > 0, "trust_people should have a positive coefficient"

    def test_odds_ratio_positive(self):
        records = _make_records_with_signal(100)
        X, y = build_feature_matrix(records, feature_names=["trust_people"])
        result = run_logistic_regression(X, y, feature_names=["trust_people"])
        for coef in result.coefficients:
            assert coef.odds_ratio > 0

    def test_cooperation_rate_in_range(self):
        records = _make_records_with_signal(100)
        X, y = build_feature_matrix(records)
        result = run_logistic_regression(X, y, feature_names=ESS_FEATURE_NAMES)
        assert 0.0 <= result.cooperation_rate <= 1.0

    def test_train_accuracy_in_range(self):
        records = _make_records_with_signal(100)
        X, y = build_feature_matrix(records)
        result = run_logistic_regression(X, y, feature_names=ESS_FEATURE_NAMES)
        assert 0.5 <= result.train_accuracy <= 1.0

    def test_rank_ordering_unique(self):
        records = _make_records_with_signal(100)
        X, y = build_feature_matrix(records)
        result = run_logistic_regression(X, y, feature_names=ESS_FEATURE_NAMES)
        ranks = [c.abs_rank for c in result.coefficients]
        assert sorted(ranks) == list(range(1, len(ESS_FEATURE_NAMES) + 1))

    def test_shape_mismatch_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="rows"):
            run_logistic_regression(X, y, feature_names=["trust_people", "risk_tolerance"])

    def test_feature_name_mismatch_raises(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        with pytest.raises(ValueError, match="columns"):
            run_logistic_regression(X, y, feature_names=["trust_people"])


# ── compute_ablation_table ────────────────────────────────────────────────────


class TestComputeAblationTable:
    def test_returns_dict_with_expected_keys(self):
        records = _make_records_with_signal(100)
        ablation = compute_ablation_table(records)
        assert set(ablation.keys()) == {"minimal", "medium", "full"}

    def test_accuracy_in_range(self):
        records = _make_records_with_signal(100)
        ablation = compute_ablation_table(records)
        for level, acc in ablation.items():
            assert 0.0 <= acc <= 1.0, f"{level} accuracy out of range: {acc}"

    def test_full_geq_minimal_accuracy(self):
        """Full profile should not perform worse than minimal (more info ≥)."""
        records = _make_records_with_signal(200)
        ablation = compute_ablation_table(records)
        # This may occasionally fail with very small N; 200 records is safe
        assert ablation["full"] >= ablation["minimal"] - 0.05

    def test_custom_levels(self):
        records = _make_records_with_signal(100)
        custom = {"just_trust": ["trust_people"]}
        ablation = compute_ablation_table(records, profile_levels=custom)
        assert "just_trust" in ablation


# ── FeatureImportanceResult helpers ──────────────────────────────────────────


class TestFeatureImportanceResult:
    def _result(self):
        records = _make_records_with_signal(150)
        X, y = build_feature_matrix(records)
        return run_logistic_regression(X, y, feature_names=ESS_FEATURE_NAMES)

    def test_top_features_length(self):
        result = self._result()
        top = result.top_features(3)
        assert len(top) == 3

    def test_top_features_sorted_by_abs(self):
        result = self._result()
        top = result.top_features(5)
        abs_coefs = [abs(fc.coefficient) for fc in top]
        assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_to_table_rows_structure(self):
        result = self._result()
        rows = result.to_table_rows()
        assert len(rows) == len(ESS_FEATURE_NAMES)
        assert all("feature" in r and "coefficient" in r and "odds_ratio" in r for r in rows)

    def test_to_table_rows_sorted_by_rank(self):
        result = self._result()
        rows = result.to_table_rows()
        ranks = [r["rank"] for r in rows]
        assert ranks == sorted(ranks)
