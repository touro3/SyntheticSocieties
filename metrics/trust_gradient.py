"""Trust-gradient sub-population validation metrics.

Phase 17 — Trust-Gradient Sub-Population Validation.

Tests whether BGF agents grounded in higher-trust ESS sub-populations produce
higher cooperation rates, validating that the grounding function Φ genuinely
transfers empirical trust signals to simulated behavior.

Central hypothesis:
    rank(ESS trust mean) ≈ rank(simulated cooperation rate)
    Measured via Spearman rank correlation (ρ), with significance threshold p < 0.10.

Statistical note on n=4 groups
──────────────────────────────
With n=4 ordered groups the exact permutation distribution for Spearman ρ has
only 4! = 24 equally-likely rank orderings.  The minimum achievable two-tailed
p-value is 2/24 ≈ 0.083 (one permutation more extreme on each tail), meaning
conventional α=0.05 is *unachievable* regardless of effect size.  The pre-
registered threshold is therefore p < 0.10.  Three complementary statistics are
reported to triangulate:

  1. Spearman ρ          — pre-registered primary test (asymptotic p)
  2. Exact permutation p — exact two-tailed p from all 4! orderings
  3. Kendall's τ-b       — rank concordance (no ties assumption)
  4. is_monotone         — strict monotonicity of simulated cooperation rates
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations as _permutations

import numpy as np
from scipy.stats import kendalltau, spearmanr


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


def _exact_spearman_permutation_p(x: np.ndarray, observed_r: float) -> float:
    """Exact two-tailed permutation p-value for Spearman ρ.

    Enumerates all n! rank orderings of y and counts the fraction whose
    Spearman ρ is as extreme (|r| ≥ |observed_r|) as the observed value.
    Only feasible for n ≤ 8 (8! = 40,320 permutations).
    """
    n = len(x)
    ranks_x = np.argsort(np.argsort(x)).astype(float)
    count = 0
    total = 0
    for perm in _permutations(range(n)):
        ranks_y = np.array(perm, dtype=float)
        # Spearman ρ = Pearson r of ranks
        r = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        if abs(r) >= abs(observed_r) - 1e-10:
            count += 1
        total += 1
    return count / total


def compute_trust_recovery_correlation(
    group_results: dict[str, dict],
    cultural_groups: list[TrustGroup] | None = None,
) -> dict:
    """Compute trust-gradient correlation between ESS trust means and simulated cooperation.

    Reports three complementary statistics to triangulate the relationship
    under the power constraint imposed by n=4 groups:

      - Spearman ρ (pre-registered primary statistic)
      - Exact permutation p-value (precise finite-sample inference)
      - Kendall's τ-b (concordance index, independent confirmation)
      - is_monotone (strict rank preservation)

    The significance threshold is p < 0.10 because with n=4 groups the minimum
    achievable two-tailed p under perfect rank agreement is 2/24 ≈ 0.083.
    Conventional α=0.05 is structurally unachievable regardless of effect size.

    Args:
        group_results: Mapping of group name → metrics dict with ``coop_rate``.
        cultural_groups: Groups to include. Defaults to :data:`TRUST_GROUPS`.

    Returns:
        dict with keys:
            spearman_r, p_value_asymptotic, p_value_exact, kendall_tau,
            kendall_p, is_monotone, n_groups, min_achievable_p,
            is_significant, interpretation.
    """
    groups = cultural_groups if cultural_groups is not None else TRUST_GROUPS

    ess_trust = np.array([g.ess_reference_mean for g in groups])
    sim_coop = np.array([float(group_results[g.name]["coop_rate"]) for g in groups])

    n = len(groups)
    # Minimum achievable two-tailed p under any ranking (1 extreme perm each tail)
    import math
    min_p = 2.0 / math.factorial(n)

    if np.std(sim_coop) < 1e-10:
        r, p_asymp = 0.0, 1.0
        tau, p_tau = 0.0, 1.0
        p_exact = 1.0
    else:
        sp = spearmanr(ess_trust, sim_coop)
        r = float(sp.statistic)
        p_asymp = float(sp.pvalue)
        p_exact = _exact_spearman_permutation_p(ess_trust, r)
        kt = kendalltau(ess_trust, sim_coop)
        tau = float(kt.statistic)
        p_tau = float(kt.pvalue)

    # Strict monotone: each step in trust produces a strictly higher coop rate
    is_mono = bool(np.all(np.diff(sim_coop) > 0))

    is_sig = bool(p_exact < 0.10 and r > 0)

    if is_sig:
        interp = (
            f"Gradient recovery confirmed: Spearman ρ={r:.3f} (exact p={p_exact:.3f}), "
            f"Kendall τ={tau:.3f} (p={p_tau:.3f}), monotone={is_mono}. "
            f"Higher-trust ESS sub-populations produce higher cooperation rates."
        )
    elif r > 0:
        interp = (
            f"Positive trend observed: Spearman ρ={r:.3f} (exact p={p_exact:.3f}), "
            f"Kendall τ={tau:.3f}, monotone={is_mono}. "
            f"Not significant at p<0.10; note minimum achievable p={min_p:.3f} at n={n}."
        )
    else:
        interp = (
            f"No gradient recovery: Spearman ρ={r:.3f} (exact p={p_exact:.3f}), "
            f"Kendall τ={tau:.3f}, monotone={is_mono}."
        )

    return {
        "spearman_r": round(r, 4),
        "p_value": round(p_exact, 4),          # canonical key; exact permutation (preferred for n≤8)
        "p_value_asymptotic": round(p_asymp, 4),
        "p_value_exact": round(p_exact, 4),
        "kendall_tau": round(tau, 4),
        "kendall_p": round(p_tau, 4),
        "is_monotone": is_mono,
        "n_groups": n,
        "min_achievable_p": round(min_p, 4),
        "is_significant": is_sig,
        "interpretation": interp,
    }
