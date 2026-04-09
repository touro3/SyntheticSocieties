"""
Distribution similarity metrics for comparing empirical vs simulated distributions.

Implements Jensen–Shannon divergence, KL divergence, and Wasserstein distance
as specified in the BGF Phase 8 evaluation metrics.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import stats


def _to_distribution(values: Iterable[float], bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw values to a normalized probability distribution via histogram."""
    arr = np.array(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("Expected at least one non-NaN value.")
    counts, bin_edges = np.histogram(arr, bins=bins, density=False)
    # Add small epsilon to avoid zero bins for KL
    probs = (counts + 1e-10) / (counts + 1e-10).sum()
    return probs, bin_edges


def jensen_shannon_divergence(
    p_values: Iterable[float],
    q_values: Iterable[float],
    bins: int = 30,
) -> float:
    """
    Compute Jensen–Shannon divergence between two distributions.

    JSD is the symmetric, bounded (0-1) version of KL divergence.
    Uses shared bin edges for fair comparison.

    Args:
        p_values: First distribution (raw values).
        q_values: Second distribution (raw values).
        bins: Number of histogram bins.

    Returns:
        JSD value in [0, 1] (base-2 logarithm).
    """
    p_arr = np.array(list(p_values), dtype=float)
    q_arr = np.array(list(q_values), dtype=float)
    p_arr = p_arr[~np.isnan(p_arr)]
    q_arr = q_arr[~np.isnan(q_arr)]

    if p_arr.size == 0 or q_arr.size == 0:
        raise ValueError("Both distributions must have at least one non-NaN value.")

    # Shared bin edges
    all_vals = np.concatenate([p_arr, q_arr])
    _, bin_edges = np.histogram(all_vals, bins=bins)

    p_counts, _ = np.histogram(p_arr, bins=bin_edges)
    q_counts, _ = np.histogram(q_arr, bins=bin_edges)

    eps = 1e-10
    p = (p_counts + eps) / (p_counts + eps).sum()
    q = (q_counts + eps) / (q_counts + eps).sum()

    return float(stats.entropy((p + q) / 2) - (stats.entropy(p) + stats.entropy(q)) / 2)


def kl_divergence(
    p_values: Iterable[float],
    q_values: Iterable[float],
    bins: int = 30,
) -> float:
    """
    Compute KL divergence D(P || Q).

    Measures how distribution P diverges from reference Q.
    Note: KL divergence is asymmetric and unbounded.

    Args:
        p_values: Distribution P (raw values).
        q_values: Reference distribution Q (raw values).
        bins: Number of histogram bins.

    Returns:
        KL divergence (non-negative float).
    """
    p_arr = np.array(list(p_values), dtype=float)
    q_arr = np.array(list(q_values), dtype=float)
    p_arr = p_arr[~np.isnan(p_arr)]
    q_arr = q_arr[~np.isnan(q_arr)]

    if p_arr.size == 0 or q_arr.size == 0:
        raise ValueError("Both distributions must have at least one non-NaN value.")

    all_vals = np.concatenate([p_arr, q_arr])
    _, bin_edges = np.histogram(all_vals, bins=bins)

    p_counts, _ = np.histogram(p_arr, bins=bin_edges)
    q_counts, _ = np.histogram(q_arr, bins=bin_edges)

    eps = 1e-10
    p = (p_counts + eps) / (p_counts + eps).sum()
    q = (q_counts + eps) / (q_counts + eps).sum()

    return float(stats.entropy(p, q))


def wasserstein_distance(
    p_values: Iterable[float],
    q_values: Iterable[float],
) -> float:
    """
    Compute Wasserstein-1 (earth mover's) distance between two distributions.

    Measures the minimum "work" needed to transform one distribution into the other.

    Args:
        p_values: First distribution (raw values).
        q_values: Second distribution (raw values).

    Returns:
        Wasserstein distance (non-negative float).
    """
    p_arr = np.array(list(p_values), dtype=float)
    q_arr = np.array(list(q_values), dtype=float)
    p_arr = p_arr[~np.isnan(p_arr)]
    q_arr = q_arr[~np.isnan(q_arr)]

    if p_arr.size == 0 or q_arr.size == 0:
        raise ValueError("Both distributions must have at least one non-NaN value.")

    return float(stats.wasserstein_distance(p_arr, q_arr))
