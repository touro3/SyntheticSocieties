"""Tests for distribution similarity metrics."""

import numpy as np
import pytest

from metrics.distribution import jensen_shannon_divergence, kl_divergence, wasserstein_distance


def test_jsd_identical_distributions():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
    jsd = jensen_shannon_divergence(a, a)
    assert jsd < 0.01, f"JSD of identical distributions should be ~0, got {jsd}"


def test_jsd_different_distributions():
    a = list(np.random.normal(0, 1, 500))
    b = list(np.random.normal(5, 1, 500))
    jsd = jensen_shannon_divergence(a, b)
    assert jsd > 0.1, f"JSD of very different distributions should be large, got {jsd}"


def test_jsd_symmetry():
    a = list(np.random.normal(0, 1, 200))
    b = list(np.random.normal(2, 1, 200))
    assert abs(jensen_shannon_divergence(a, b) - jensen_shannon_divergence(b, a)) < 0.01


def test_jsd_bounded():
    a = list(np.random.normal(0, 1, 200))
    b = list(np.random.normal(10, 1, 200))
    jsd = jensen_shannon_divergence(a, b)
    assert 0 <= jsd <= 1.0, f"JSD should be in [0, 1], got {jsd}"


def test_kl_identical():
    a = list(np.random.normal(0, 1, 500))
    kl = kl_divergence(a, a)
    assert kl < 0.01, f"KL of identical distributions should be ~0, got {kl}"


def test_kl_asymmetric():
    a = list(np.random.normal(0, 1, 500))
    b = list(np.random.normal(3, 1, 500))
    kl_ab = kl_divergence(a, b)
    kl_ba = kl_divergence(b, a)
    # KL is asymmetric, both should be positive but not necessarily equal
    assert kl_ab > 0
    assert kl_ba > 0


def test_wasserstein_identical():
    a = [1, 2, 3, 4, 5] * 20
    wd = wasserstein_distance(a, a)
    assert wd < 0.01, f"Wasserstein of identical distributions should be ~0, got {wd}"


def test_wasserstein_shifted():
    a = list(np.random.normal(0, 1, 500))
    b = list(np.random.normal(5, 1, 500))
    wd = wasserstein_distance(a, b)
    assert 4.0 < wd < 6.0, f"Wasserstein of shifted normals should be ~5, got {wd}"


def test_wasserstein_non_negative():
    a = list(np.random.uniform(0, 1, 200))
    b = list(np.random.uniform(0, 1, 200))
    assert wasserstein_distance(a, b) >= 0


def test_empty_raises():
    with pytest.raises(ValueError):
        jensen_shannon_divergence([], [1, 2, 3])
    with pytest.raises(ValueError):
        kl_divergence([1, 2], [])
    with pytest.raises(ValueError):
        wasserstein_distance([], [])
