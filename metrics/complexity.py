"""Phase transition detection and power law fitting.

Emergent complexity analysis.

Provides:
  - Sigmoid fitting for phase transition detection in parameter sweeps
  - Power law fitting via Clauset et al. 2009 MLE
  - Sweep analysis aggregating multiple metrics

No new dependencies: uses scipy.optimize.curve_fit (already available)
and numpy MLE for power law (15 lines, no powerlaw package needed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest

# ── Sigmoid ──────────────────────────────────────────────────────────────


def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Standard logistic sigmoid.

    f(x) = L / (1 + exp(-k * (x - x0))) + b

    Args:
        x: Input array.
        L: Curve amplitude (maximum - minimum).
        k: Steepness (higher = sharper transition).
        x0: Inflection point (midpoint of transition).
        b: Vertical offset (baseline).
    """
    return L / (1.0 + np.exp(-k * (x - x0))) + b


# ── Phase transition fitting ─────────────────────────────────────────────


def fit_phase_transition(
    sweep_param: np.ndarray,
    metric_values: np.ndarray,
) -> dict[str, Any]:
    """Fit a sigmoid to sweep data and extract phase transition parameters.

    Args:
        sweep_param: Parameter values (e.g., bad_apple_fraction 0.0-0.4).
        metric_values: Observed metric at each parameter value.

    Returns:
        {
            'inflection_point': x0 where transition occurs,
            'steepness': k parameter,
            'r_squared': goodness of fit,
            'is_transition': True if r^2 > 0.8 and |steepness| > 3.0,
            'fit_params': {'L', 'k', 'x0', 'b'},
        }
    """
    x = np.asarray(sweep_param, dtype=float)
    y = np.asarray(metric_values, dtype=float)

    if len(x) < 5:
        return {
            "inflection_point": float(np.nan),
            "steepness": 0.0,
            "r_squared": 0.0,
            "is_transition": False,
            "fit_params": {"L": 0.0, "k": 0.0, "x0": 0.0, "b": 0.0},
        }

    # Initial guesses
    y_range = float(y.max() - y.min())
    if y_range < 1e-10:
        return {
            "inflection_point": float(np.nan),
            "steepness": 0.0,
            "r_squared": 0.0,
            "is_transition": False,
            "fit_params": {"L": 0.0, "k": 0.0, "x0": float(x.mean()), "b": float(y.mean())},
        }

    p0 = [y_range, 10.0, float(x.mean()), float(y.min())]
    bounds = (
        [-2 * abs(y_range), -200.0, float(x.min()) - 1, float(y.min()) - abs(y_range)],
        [2 * abs(y_range), 200.0, float(x.max()) + 1, float(y.max()) + abs(y_range)],
    )

    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return {
            "inflection_point": float(np.nan),
            "steepness": 0.0,
            "r_squared": 0.0,
            "is_transition": False,
            "fit_params": {"L": 0.0, "k": 0.0, "x0": 0.0, "b": 0.0},
        }

    L, k, x0, b = popt
    y_pred = sigmoid(x, *popt)

    # R-squared
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    is_transition = r_squared > 0.8 and abs(k) > 3.0

    return {
        "inflection_point": float(x0),
        "steepness": float(k),
        "r_squared": r_squared,
        "is_transition": is_transition,
        "fit_params": {"L": float(L), "k": float(k), "x0": float(x0), "b": float(b)},
    }


# ── Power law fitting (Clauset et al. 2009 MLE) ─────────────────────────


def fit_power_law(
    values: np.ndarray,
    xmin: float | None = None,
) -> dict[str, Any]:
    """Fit power law to a distribution via Maximum Likelihood Estimation.

    Uses the discrete MLE estimator from Clauset, Shalizi & Newman (2009):
        alpha = 1 + n * (sum(ln(x_i / xmin)))^-1

    Args:
        values: Observed values (e.g., wealth distribution).
        xmin: Minimum value for fit. If None, uses the minimum of values.

    Returns:
        {
            'alpha': power law exponent,
            'xmin': minimum value used,
            'ks_statistic': Kolmogorov-Smirnov goodness of fit,
            'p_value': p-value from KS test,
            'is_power_law': True if p > 0.1 and alpha in [1.5, 3.5],
            'reliable': True if n >= 50,
        }
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return _empty_power_law_result()

    if xmin is None:
        xmin = float(arr.min())

    # Filter to values >= xmin
    tail = arr[arr >= xmin]
    n = len(tail)

    if n < 2 or xmin <= 0:
        return _empty_power_law_result(xmin=xmin, reliable=n >= 50)

    # MLE: alpha = 1 + n / sum(ln(x_i / xmin))
    log_sum = np.sum(np.log(tail / xmin))
    if log_sum <= 0:
        return _empty_power_law_result(xmin=xmin, reliable=n >= 50)

    alpha = 1.0 + n / log_sum

    # KS test against theoretical power law CDF
    # CDF of power law: F(x) = 1 - (x/xmin)^(-(alpha-1))
    def power_law_cdf(x: np.ndarray) -> np.ndarray:
        return 1.0 - (x / xmin) ** (-(alpha - 1.0))

    try:
        ks_stat, p_value = kstest(tail, power_law_cdf)
    except Exception:
        ks_stat, p_value = 1.0, 0.0

    is_power_law = p_value > 0.1 and 1.5 <= alpha <= 3.5
    reliable = n >= 50

    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "is_power_law": is_power_law and reliable,
        "reliable": reliable,
    }


def _empty_power_law_result(xmin: float = 0.0, reliable: bool = False) -> dict[str, Any]:
    return {
        "alpha": float(np.nan),
        "xmin": xmin,
        "ks_statistic": float(np.nan),
        "p_value": 0.0,
        "is_power_law": False,
        "reliable": reliable,
    }


# ── Sweep analysis ───────────────────────────────────────────────────────


def analyze_sweep_results(
    sweep_values: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Analyze multiple metrics across a parameter sweep.

    Args:
        sweep_values: Parameter values (e.g., bad_apple_fraction 0.0-0.4).
        metrics: {metric_name: values_array}. Each array must have the
            same length as sweep_values.

    Returns:
        {metric_name: phase_transition_result} for each metric.
    """
    results = {}
    x = np.asarray(sweep_values, dtype=float)

    for name, values in metrics.items():
        y = np.asarray(values, dtype=float)
        results[name] = fit_phase_transition(x, y)

    return results
