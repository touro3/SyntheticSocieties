"""Trust-gradient sub-population validation metrics.

Phase 17 — Trust-Gradient Sub-Population Validation.

Tests whether BGF agents grounded in higher-trust ESS sub-populations produce
higher cooperation rates, validating that the grounding function Φ genuinely
transfers empirical trust signals to simulated behavior.

Central hypothesis:
    rank(ESS trust mean) ≈ rank(simulated cooperation rate)
    Measured via Spearman rank correlation (r), with significance threshold p < 0.10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class TrustGroup:
    """Defines a trust-level sub-population for gradient validation.

    Attributes:
        name: Human-readable group label (e.g., "High-Trust").
        band: SocietySpec trust_people_band value used to filter the ESS cohort.
        trust_range: (lo, hi) normalized [0,1] trust interval for this band.
        ess_reference_mean: Empirical mean trust for this band in the current
            ESS parquet (Austria sample; precomputed from the cleaned data).
    """

    name: str
    band: str
    trust_range: tuple[float, float]
    ess_reference_mean: float


# ── Canonical group definitions ───────────────────────────────────────────────
# Reference means computed from data/ess_clean.parquet (Austria, ESS Round 11).
# Ordered from lowest to highest trust so that a positive Spearman r indicates
# correct gradient recovery.

TRUST_GROUPS: list[TrustGroup] = [
    TrustGroup(
        name="Low-Trust",
        band="low",
        trust_range=(0.2, 0.4),
        ess_reference_mean=0.267,
    ),
    TrustGroup(
        name="Moderate-Trust",
        band="moderate",
        trust_range=(0.4, 0.6),
        ess_reference_mean=0.467,
    ),
    TrustGroup(
        name="High-Trust",
        band="high",
        trust_range=(0.6, 0.8),
        ess_reference_mean=0.657,
    ),
    TrustGroup(
        name="Very-High-Trust",
        band="very_high",
        trust_range=(0.8, 1.0),
        ess_reference_mean=0.839,
    ),
]


def compute_trust_gradient(
    group_results: dict[str, dict],
    trust_groups: list[TrustGroup] | None = None,
) -> dict[str, float]:
    """Extract the cooperation rate for each trust group.

    Args:
        group_results: Mapping of group name → metrics dict.
            Each dict must contain at least ``{"coop_rate": float, ...}``.
        trust_groups: Groups to include. Defaults to :data:`TRUST_GROUPS`.

    Returns:
        Mapping of group name → simulated cooperation rate, preserving the
        ordering of ``trust_groups`` (ascending trust).

    Raises:
        KeyError: If a group name from ``trust_groups`` is absent from
            ``group_results``.
        ValueError: If ``group_results`` is empty.
    """
    if not group_results:
        raise ValueError("group_results must not be empty")

    groups = trust_groups if trust_groups is not None else TRUST_GROUPS

    gradient: dict[str, float] = {}
    for g in groups:
        metrics = group_results[g.name]  # raises KeyError if missing
        gradient[g.name] = float(metrics["coop_rate"])
    return gradient


def compute_trust_recovery_correlation(
    group_results: dict[str, dict],
    cultural_groups: list[TrustGroup] | None = None,
) -> dict:
    """Compute Spearman rank correlation between ESS trust and simulated cooperation.

    This is the primary validation metric for Phase 17. A positive, significant
    correlation (r > 0, p < 0.10) provides evidence that the grounding function Φ
    transfers empirical trust signals to simulated behavioral outcomes.

    The significance threshold is set at p < 0.10 (rather than the conventional
    0.05) because with n=4 groups the exact permutation test has only 24 possible
    rank orderings and the minimum achievable p-value is 1/24 ≈ 0.042.

    Args:
        group_results: Mapping of group name → metrics dict with ``coop_rate``.
        cultural_groups: Groups to include. Defaults to :data:`TRUST_GROUPS`.

    Returns:
        dict with keys:
            - ``spearman_r``: Spearman correlation coefficient ∈ [-1, 1].
            - ``p_value``: Two-tailed p-value for the null hypothesis r = 0.
            - ``n_groups``: Number of groups compared.
            - ``is_significant``: True when p_value < 0.10.
            - ``interpretation``: Human-readable summary of the finding.
    """
    groups = cultural_groups if cultural_groups is not None else TRUST_GROUPS

    ess_trust = np.array([g.ess_reference_mean for g in groups])
    sim_coop = np.array([float(group_results[g.name]["coop_rate"]) for g in groups])

    n = len(groups)

    if np.std(sim_coop) < 1e-10:
        # Constant sequence — correlation undefined
        r, p = 0.0, 1.0
    else:
        result = spearmanr(ess_trust, sim_coop)
        r = float(result.statistic)
        p = float(result.pvalue)

    is_sig = bool(p < 0.10 and r > 0)

    if is_sig:
        interp = (
            f"Positive gradient recovery confirmed (r={r:.3f}, p={p:.3f}). "
            f"Higher-trust ESS sub-populations produce higher cooperation rates, "
            f"supporting the grounding hypothesis."
        )
    elif r > 0:
        interp = (
            f"Positive trend observed (r={r:.3f}, p={p:.3f}) but not statistically "
            f"significant at p<0.10. More seeds or agents may be needed."
        )
    else:
        interp = (
            f"No positive gradient recovery (r={r:.3f}, p={p:.3f}). "
            f"Trust grounding did not produce a cooperation gradient in this run."
        )

    return {
        "spearman_r": r,
        "p_value": p,
        "n_groups": n,
        "is_significant": is_sig,
        "interpretation": interp,
    }
