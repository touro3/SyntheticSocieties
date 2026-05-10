from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini coefficient of a wealth distribution.

    Uses the sorted-index formula: G = (2 * Σ(i * x_i) / (n * Σx_i)) - (n+1)/n.
    Returns 0.0 for a perfectly equal distribution and approaches 1.0 for maximum inequality.

    Args:
        values: Non-negative wealth values for each agent.

    Returns:
        Gini coefficient in [0, 1].
    """
    x = np.array(list(values), dtype=float)

    if x.size == 0:
        raise ValueError("Expected at least one value.")

    if np.any(x < 0):
        raise ValueError("Gini coefficient is not defined for negative values in this implementation.")

    if np.allclose(x, 0):
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * x_sorted) / (n * np.sum(x_sorted))) - (n + 1) / n
    return float(gini)


def lorenz_curve(values: Iterable[float]) -> dict[str, list[float]]:
    """Compute the Lorenz curve for a wealth distribution.

    Args:
        values: Non-negative wealth values for each agent.

    Returns:
        Dict with keys ``"population"`` and ``"wealth"``, each a list of
        cumulative share values from 0.0 to 1.0 (suitable for plotting).
    """
    x = np.array(list(values), dtype=float)

    if x.size == 0:
        raise ValueError("Expected at least one value.")

    if np.any(x < 0):
        raise ValueError("Lorenz curve is not defined for negative values in this implementation.")

    x_sorted = np.sort(x)
    cumulative = np.cumsum(x_sorted)
    total = cumulative[-1] if cumulative.size > 0 else 0.0

    if total == 0:
        cumulative_share = np.zeros_like(cumulative, dtype=float)
    else:
        cumulative_share = cumulative / total

    population_share = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    population = np.concatenate([[0.0], population_share])
    wealth = np.concatenate([[0.0], cumulative_share])

    return {
        "population_share": population.tolist(),
        "value_share": wealth.tolist(),
    }
