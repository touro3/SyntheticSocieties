"""Statistical inference utilities — Phase 28.1.

Provides:
  - Benjamini-Hochberg FDR correction for multiple comparisons
  - Bootstrap confidence intervals (percentile method)
  - Standardised metric reporter: value ± [lower, upper]

These utilities are used across all BGF metrics to ensure statistical rigor
when reporting comparisons across seeds, conditions, or sub-populations.
"""

from __future__ import annotations

import math
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
    bootstrap_stats = np.array([fn(rng.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)])

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


# ── Bias-Corrected Accelerated (BCa) Bootstrap ───────────────────────────────


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation; adequate for CI endpoints.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def bca_ci(
    x,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float]:
    """Bias-corrected accelerated (BCa) bootstrap CI for the mean.

    BCa adjusts the percentile bootstrap for bias (``z0``) and skew
    (acceleration ``a`` via jackknife), so on skewed seed distributions the
    interval is asymmetric about the point estimate — the correct behaviour
    for headline metrics reported across a small number of seeds.

    Args:
        x: Observed values (e.g. one metric across seeds).
        alpha: 1 - coverage (0.05 → 95% CI).
        n_boot: Number of bootstrap resamples.
        rng: ``None`` → a fresh ``np.random.default_rng(42)`` (paper seed
            convention, deterministic per call); an ``int`` → seeded
            generator; an existing ``np.random.Generator`` → used as-is and
            advanced (lets a caller share one stream across many calls).

    Returns:
        ``(lower, upper)``. Returns ``(nan, nan)`` when ``len(x) < 2``
        (a CI is undefined for fewer than two observations).
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(42)
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    boot = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    theta = x.mean()
    # bias-correction z0
    prop = np.mean(boot < theta)
    prop = min(max(prop, 1.0 / n_boot), 1 - 1.0 / n_boot)
    z0 = _norm_ppf(prop)
    # acceleration via jackknife
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
    a1 = _norm_cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = _norm_cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    lo = np.quantile(boot, max(0.0, min(1.0, a1)))
    hi = np.quantile(boot, max(0.0, min(1.0, a2)))
    return float(lo), float(hi)


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


# ── Effect Size ──────────────────────────────────────────────────────────────


def cohens_d(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """Cohen's d effect size: (mean_a - mean_b) / pooled_std.

    Uses the pooled standard deviation (equal-variance form).  For n < 50
    consider Hedges' g (bias-corrected), but Cohen's d is standard for
    simulation paper comparisons.

    Returns:
        Positive d means group_a > group_b.  Conventional thresholds:
        small |d|=0.2, medium |d|=0.5, large |d|=0.8.
    """
    a, b = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Both groups must have at least 2 observations.")
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def hedges_g(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """Hedges' g: bias-corrected effect size for small samples.

    Preferred over Cohen's d when n < 50 (typical for multi-seed simulation
    experiments).  The correction factor J(df) ≈ 1 − 3/(4·df − 1) removes
    the positive bias of d in small samples.

    Args:
        group_a: Observations from condition A.
        group_b: Observations from condition B.

    Returns:
        Hedges' g (positive when group_a > group_b).  Magnitude thresholds
        are the same as Cohen's d: small 0.2, medium 0.5, large 0.8.
    """
    d = cohens_d(group_a, group_b)
    n = len(group_a) + len(group_b)
    df = n - 2
    if df <= 0:
        return d
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * j


# ── Non-Parametric Significance ───────────────────────────────────────────────


def mann_whitney_u(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alternative: str = "two-sided",
) -> dict:
    """Mann-Whitney U test (non-parametric alternative to t-test).

    Does not assume normality — appropriate for simulation metrics whose
    distributions are often skewed (Gini, cooperation rate, wealth).

    Args:
        group_a: Observations from condition A.
        group_b: Observations from condition B.
        alternative: "two-sided", "greater", or "less".

    Returns:
        Dict with keys: u_statistic, p_value, significant (at α=0.05).
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise ImportError("scipy is required for mann_whitney_u. Install with: pip install scipy") from exc

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        raise ValueError("Mann-Whitney U requires at least 3 observations per group.")

    u_stat, p_val = mannwhitneyu(a, b, alternative=alternative)
    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "alternative": alternative,
        "n_a": len(a),
        "n_b": len(b),
    }


# ── Power Analysis ────────────────────────────────────────────────────────────


def min_seeds_for_power(
    effect_size_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Estimate minimum seeds (n per group) needed for a two-sample t-test.

    Uses the analytic formula: n = 2 * ((z_alpha/2 + z_beta) / d)^2.
    This is a simple closed-form approximation; for exact results use
    ``statsmodels.stats.power.TTestIndPower``.

    Args:
        effect_size_d: Expected Cohen's d (use 0.5 for medium, 0.8 for large).
        alpha: Type-I error rate (default 0.05, two-tailed).
        power: Desired power 1-β (default 0.80).

    Returns:
        Minimum n per group (seeds), rounded up.

    Example:
        >>> min_seeds_for_power(0.5)  # medium effect
        64
        >>> min_seeds_for_power(0.8)  # large effect
        26
    """
    if effect_size_d <= 0:
        raise ValueError("effect_size_d must be positive.")
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed
    z_beta = norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size_d) ** 2
    return int(np.ceil(n))


def power_report(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Full power + significance report for two groups.

    Combines Cohen's d, Mann-Whitney U, bootstrap CIs, and minimum seeds
    needed for 80% power at the observed effect size.

    Returns dict with: cohens_d, interpretation, mann_whitney, min_seeds_80,
    ci_a, ci_b.
    """
    d = cohens_d(group_a, group_b)
    g = hedges_g(group_a, group_b)
    mw = mann_whitney_u(group_a, group_b, alternative="two-sided")
    ci_a = report_metric(list(group_a))
    ci_b = report_metric(list(group_b))

    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"

    try:
        min_seeds = min_seeds_for_power(abs(d), alpha=alpha) if d != 0.0 else None
    except Exception:
        min_seeds = None

    return {
        "cohens_d": round(d, 4),
        "hedges_g": round(g, 4),
        "interpretation": interp,
        "mann_whitney": mw,
        "min_seeds_80pct_power": min_seeds,
        "ci_a": ci_a,
        "ci_b": ci_b,
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
