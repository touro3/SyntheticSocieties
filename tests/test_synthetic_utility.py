"""Tests for metrics/synthetic_utility.py."""

from __future__ import annotations

import numpy as np
import pytest

from metrics.synthetic_utility import (
    MAX_UTILITY_GAP,
    TRUST_THRESHOLD,
    TSTRResult,
    build_synthetic_dataset,
    profile_to_feature_vector,
    profile_to_label,
    tstr_benchmark,
    utility_report,
)
from tests.conftest import make_profile


# ── profile_to_feature_vector ─────────────────────────────────────────────────

def test_feature_vector_length():
    p = make_profile(age=35, income=500.0)
    vec = profile_to_feature_vector(p)
    assert len(vec) == 4


def test_feature_vector_values_in_unit_interval():
    p = make_profile(age=35, income=500.0)
    vec = profile_to_feature_vector(p)
    for v in vec:
        assert 0.0 <= v <= 1.0, f"Feature out of [0,1]: {v}"


def test_feature_vector_from_dict():
    d = {"age": 40, "income_decile": 6, "education_years": 12, "social_activity": 0.6}
    vec = profile_to_feature_vector(d)
    assert len(vec) == 4
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_feature_vector_none_values_handled():
    # Profile with missing optional fields
    p = make_profile(age=50)
    vec = profile_to_feature_vector(p)
    assert len(vec) == 4
    assert not any(np.isnan(v) for v in vec)


def test_feature_vector_age_normalisation():
    young = make_profile(age=15)
    old = make_profile(age=90)
    young_vec = profile_to_feature_vector(young)
    old_vec = profile_to_feature_vector(old)
    assert young_vec[0] < old_vec[0]


# ── profile_to_label ──────────────────────────────────────────────────────────

def test_label_high_trust():
    p = make_profile(trust_people=0.8)
    assert profile_to_label(p) == 1


def test_label_low_trust():
    p = make_profile(trust_people=0.2)
    assert profile_to_label(p) == 0


def test_label_at_threshold():
    p = make_profile(trust_people=TRUST_THRESHOLD)
    assert profile_to_label(p) == 1  # >= threshold → 1


def test_label_from_dict():
    assert profile_to_label({"trust_people": 0.7}) == 1
    assert profile_to_label({"trust_people": 0.3}) == 0


def test_label_default_when_missing():
    # trust_people defaults to 0.5 → label = 1
    label = profile_to_label({})
    assert label in {0, 1}


# ── build_synthetic_dataset ───────────────────────────────────────────────────

def _make_profiles(n: int, trust: float = 0.5) -> list:
    return [make_profile(agent_id=f"a_{i}", trust_people=trust, age=30 + i % 40)
            for i in range(n)]


def test_build_synthetic_dataset_shape():
    profiles = _make_profiles(20)
    X, y = build_synthetic_dataset(profiles)
    assert X.shape == (20, 4)
    assert y.shape == (20,)


def test_build_synthetic_dataset_dtypes():
    X, y = build_synthetic_dataset(_make_profiles(10))
    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_build_synthetic_dataset_labels_binary():
    X, y = build_synthetic_dataset(_make_profiles(15))
    assert set(y.tolist()).issubset({0, 1})


def test_build_synthetic_dataset_empty():
    X, y = build_synthetic_dataset([])
    assert X.shape[0] == 0
    assert y.shape[0] == 0


# ── tstr_benchmark ────────────────────────────────────────────────────────────

def _make_dataset(n: int, signal_strength: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Create a dataset with a learnable signal between features and labels."""
    rng = np.random.default_rng(42)
    X = rng.random((n, 4)).astype(np.float32)
    # Label positively correlated with first feature (age_norm proxy)
    y = (X[:, 0] * signal_strength + rng.random(n) * (1 - signal_strength) > 0.5).astype(np.int32)
    return X, y


def test_tstr_returns_result():
    synth_X, synth_y = _make_dataset(200)
    real_X, real_y = _make_dataset(200)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    assert isinstance(result, TSTRResult)


def test_tstr_accuracy_in_unit_interval():
    synth_X, synth_y = _make_dataset(150)
    real_X, real_y = _make_dataset(150)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    assert 0.0 <= result.tstr_accuracy <= 1.0
    assert 0.0 <= result.trtr_accuracy <= 1.0


def test_tstr_auc_in_unit_interval():
    synth_X, synth_y = _make_dataset(150)
    real_X, real_y = _make_dataset(150)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    assert 0.0 <= result.tstr_auc <= 1.0
    assert 0.0 <= result.trtr_auc <= 1.0


def test_tstr_sample_counts():
    synth_X, synth_y = _make_dataset(100)
    real_X, real_y = _make_dataset(80)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y, test_fraction=0.25)
    assert result.n_synthetic_train == 100
    assert result.n_real_test == pytest.approx(20, abs=2)
    assert result.n_real_train == pytest.approx(60, abs=2)


def test_tstr_utility_gap_same_distribution():
    """When synthetic = real distribution, utility gap should be small."""
    rng = np.random.default_rng(0)
    X = rng.random((500, 4)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    result = tstr_benchmark(X, y, X.copy(), y.copy())
    # Same data → gap should be very small (near 0)
    assert abs(result.utility_gap_accuracy) < 0.10


def test_tstr_label_balance_computed():
    rng = np.random.default_rng(5)
    # Synthetic: 90% positive (high trust)
    synth_X = rng.random((100, 4)).astype(np.float32)
    synth_y = (rng.random(100) < 0.9).astype(np.int32)
    # Real: 40% positive
    real_X = rng.random((100, 4)).astype(np.float32)
    real_y = (rng.random(100) < 0.4).astype(np.int32)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    assert result.synthetic_label_balance > 0.80
    assert result.real_label_balance < 0.55


def test_tstr_raises_for_small_synthetic():
    synth_X = np.random.rand(5, 4).astype(np.float32)
    synth_y = np.ones(5, dtype=np.int32)
    real_X = np.random.rand(50, 4).astype(np.float32)
    real_y = np.ones(50, dtype=np.int32)
    with pytest.raises(ValueError, match="too small"):
        tstr_benchmark(synth_X, synth_y, real_X, real_y)


def test_tstr_raises_for_small_real():
    synth_X = np.random.rand(50, 4).astype(np.float32)
    synth_y = np.ones(50, dtype=np.int32)
    real_X = np.random.rand(3, 4).astype(np.float32)
    real_y = np.ones(3, dtype=np.int32)
    with pytest.raises(ValueError, match="too small"):
        tstr_benchmark(synth_X, synth_y, real_X, real_y)


def test_tstr_passes_threshold_with_faithful_synthetic():
    """A synthetic dataset faithfully replicating the real distribution should pass."""
    rng = np.random.default_rng(7)
    n = 300
    real_X = rng.random((n, 4)).astype(np.float32)
    real_y = (real_X[:, 0] * 0.9 + rng.random(n) * 0.1 > 0.5).astype(np.int32)
    # Synthetic: same generator, different seed
    rng2 = np.random.default_rng(99)
    synth_X = rng2.random((n, 4)).astype(np.float32)
    synth_y = (synth_X[:, 0] * 0.9 + rng2.random(n) * 0.1 > 0.5).astype(np.int32)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    assert result.passes_utility_threshold, (
        f"Faithful synthetic did not pass threshold: gap={result.utility_gap_accuracy}"
    )


def test_tstr_utility_gap_calculation():
    synth_X, synth_y = _make_dataset(200)
    real_X, real_y = _make_dataset(200)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    expected_gap = round(result.trtr_accuracy - result.tstr_accuracy, 4)
    assert abs(result.utility_gap_accuracy - expected_gap) < 1e-4


# ── utility_report ─────────────────────────────────────────────────────────────

def test_utility_report_returns_string():
    synth_X, synth_y = _make_dataset(100)
    real_X, real_y = _make_dataset(100)
    result = tstr_benchmark(synth_X, synth_y, real_X, real_y)
    report = utility_report(result)
    assert isinstance(report, str)
    assert "TSTR" in report
    assert "TRTR" in report


def test_utility_report_shows_pass():
    # Build a result that will pass
    result = TSTRResult(
        tstr_accuracy=0.72, tstr_auc=0.78,
        trtr_accuracy=0.74, trtr_auc=0.80,
        utility_gap_accuracy=0.02, utility_gap_auc=0.02,
        passes_utility_threshold=True,
        n_synthetic_train=200, n_real_train=140, n_real_test=60,
        synthetic_label_balance=0.50, real_label_balance=0.48,
    )
    report = utility_report(result)
    assert "PASS" in report


def test_utility_report_shows_fail():
    result = TSTRResult(
        tstr_accuracy=0.55, tstr_auc=0.58,
        trtr_accuracy=0.75, trtr_auc=0.80,
        utility_gap_accuracy=0.20, utility_gap_auc=0.22,
        passes_utility_threshold=False,
        n_synthetic_train=200, n_real_train=140, n_real_test=60,
        synthetic_label_balance=0.40, real_label_balance=0.50,
    )
    report = utility_report(result)
    assert "FAIL" in report
