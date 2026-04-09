"""Statistical inference utilities — Phase 28.1.

Provides:
  - Benjamini-Hochberg FDR correction for multiple comparisons
  - Bootstrap confidence intervals (percentile method)
  - Standardised metric reporter: value ± [lower, upper]

These utilities are used across all BGF metrics to ensure statistical rigor
when reporting comparisons across seeds, conditions, or sub-populations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# ── Benjamini-Hochberg FDR Correction ────────────────────────────────────────


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg (1995) FDR correction for multiple comparisons.

    Computes adjusted p-values using the step-up BH procedure, controlling
    the expected False Discovery Rate at level ``alpha``.

    Args:
        p_values: Raw p-values from independent or positively-correlated tests.
        alpha: Target FDR level (default 0.05).

    Returns:
        Tuple of (adjusted_pvalues, rejected) where:
          - adjusted_pvalues[i] = min(p[i] * m / rank[i], 1.0)
          - rejected[i] = True when adjusted_pvalues[i] ≤ alpha

    Raises:
        ValueError: If p_values is empty or alpha is not in (0, 1).

    Example:
        >>> p = [0.001, 0.02, 0.04, 0.10, 0.50]
        >>> adj, rej = benjamini_hochberg(p)
        >>> rej
        [True, True, True, False, False]
    """
    if not p_values:
        raise ValueError("p_values must not be empty.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    p = np.asarray(p_values, dtype=float)
    m = len(p)

    # Sort ascending to get ranks
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]

    # BH step-up adjusted p-values (cumulative min from the right)
    adjusted = sorted_p * m / (np.arange(1, m + 1))
    # Enforce monotonicity: each adjusted p ≤ next one
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(m)
    result[sorted_indices] = adjusted

    rejected = [bool(result[i] <= alpha) for i in range(m)]
    return result.tolist(), rejected


# ── Bootstrap Confidence Intervals ───────────────────────────────────────────


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Uses the percentile method: resample ``values`` with replacement
    ``n_bootstrap`` times, compute ``stat_fn`` on each resample, and return
    the (alpha/2, 1-alpha/2) quantiles of the bootstrap distribution.

    Args:
        values: Observed data (list or 1-D array).
        stat_fn: Scalar-valued function of a 1-D array. Defaults to
            ``np.mean``.
        n_bootstrap: Number of bootstrap resamples (default 2000).
        confidence: Coverage level, e.g. 0.95 for 95% CI.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple (point_estimate, lower, upper).

    Raises:
        ValueError: If values is empty or confidence is not in (0, 1).
    """
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    fn = stat_fn if stat_fn is not None else np.mean
    arr = np.asarray(values, dtype=float)
    point = float(fn(arr))

    rng = np.random.default_rng(random_state)
    n = len(arr)
    bootstrap_stats = np.array([
        fn(rng.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


# ── Metric Reporter ───────────────────────────────────────────────────────────


def report_metric(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    decimals: int = 4,
    random_state: int = 42,
) -> dict:
    """Compute a statistic with bootstrap CI and format for reporting.

    Returns a dict with keys: ``value``, ``lower``, ``upper``, ``ci_str``.
    The ``ci_str`` is the paper-ready string ``"value [lower, upper]"``.

    Args:
        values: Observed data.
        stat_fn: Statistic function (default: mean).
        n_bootstrap: Bootstrap resamples.
        confidence: CI coverage level.
        decimals: Rounding precision for the output string.
        random_state: Random seed.

    Returns:
        Dict with keys: value, lower, upper, ci_str, n.

    Example:
        >>> r = report_metric([0.61, 0.58, 0.64])
        >>> r["ci_str"]
        '0.6100 [0.5800, 0.6400]'
    """
    point, lower, upper = bootstrap_ci(
        values,
        stat_fn=stat_fn,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_state=random_state,
    )
    fmt = f".{decimals}f"
    ci_str = f"{point:{fmt}} [{lower:{fmt}}, {upper:{fmt}}]"
    return {
        "value": round(point, decimals),
        "lower": round(lower, decimals),
        "upper": round(upper, decimals),
        "ci_str": ci_str,
        "n": len(values),
        "confidence": confidence,
    }


def apply_fdr_to_results(
    named_pvalues: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Apply BH FDR correction to a dict of named p-values.

    Args:
        named_pvalues: Mapping of test name → raw p-value.
        alpha: FDR threshold.

    Returns:
        Dict mapping test name → {"raw_p", "adjusted_p", "rejected"}.
    """
    names = list(named_pvalues.keys())
    raw_ps = [named_pvalues[n] for n in names]
    adj_ps, rejected = benjamini_hochberg(raw_ps, alpha=alpha)

    return {
        name: {
            "raw_p": raw_ps[i],
            "adjusted_p": adj_ps[i],
            "rejected": rejected[i],
        }
        for i, name in enumerate(names)
    }
