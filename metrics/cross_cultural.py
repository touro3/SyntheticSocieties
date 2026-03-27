"""Cross-cultural validation metrics — Phase 27.

Tests whether BGF agents grounded in higher-trust ESS country clusters
produce higher cooperation rates, recovering the empirical cross-cultural
trust gradient documented in ESS-11.

Central hypothesis:
    rank(ESS cluster trust mean) ≈ rank(simulated cooperation rate)
    Pearson r > 0 and Spearman ρ > 0, significance threshold p < 0.10.

Design note:
    The local ESS parquet contains only Austrian microdata. Each cluster
    simulation filters the Austrian sample by trust_people_band (high /
    moderate / low) to approximate the cluster's trust profile. Published
    ESS-11 country-level means serve as the empirical benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class ClusterSimResult:
    """Simulation result for a single ESS country cluster.

    Attributes:
        cluster_name: One of "nordic", "southern", "eastern".
        ess_mean_trust: ESS-11 published mean interpersonal trust [0, 1].
        simulated_cooperation_rate: Fraction of cooperate actions in simulation.
        simulated_gini: Gini coefficient at end of simulation.
        n_agents: Number of agents in the simulation run.
        n_rounds: Number of simulation rounds.
    """

    cluster_name: str
    ess_mean_trust: float
    simulated_cooperation_rate: float
    simulated_gini: float
    n_agents: int
    n_rounds: int


@dataclass
class CrossCulturalResult:
    """Aggregated cross-cultural validation result.

    Attributes:
        cluster_results: Per-cluster simulation results.
        pearson_r: Pearson correlation between ESS trust and cooperation rate.
        pearson_p: Two-sided p-value for Pearson r.
        spearman_rho: Spearman rank correlation.
        spearman_p: Two-sided p-value for Spearman ρ.
        gradient_recovered: True iff spearman_rho > 0 and spearman_p < 0.10.
    """

    cluster_results: list[ClusterSimResult]
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    gradient_recovered: bool


def compute_cross_cultural_correlation(
    cluster_results: list[ClusterSimResult],
) -> CrossCulturalResult:
    """Compute Pearson r and Spearman ρ between ESS trust and simulated cooperation.

    Args:
        cluster_results: At least 3 ClusterSimResult instances.

    Returns:
        CrossCulturalResult with both correlation coefficients and p-values.

    Raises:
        ValueError: If fewer than 3 clusters are provided.
    """
    if len(cluster_results) < 3:
        raise ValueError(
            f"Need at least 3 clusters for meaningful correlation; got {len(cluster_results)}."
        )

    trust_means = np.array([r.ess_mean_trust for r in cluster_results])
    coop_rates = np.array([r.simulated_cooperation_rate for r in cluster_results])

    # Degenerate case: all cooperation rates identical — no signal to correlate.
    if np.all(coop_rates == coop_rates[0]):
        return CrossCulturalResult(
            cluster_results=cluster_results,
            pearson_r=0.0,
            pearson_p=1.0,
            spearman_rho=0.0,
            spearman_p=1.0,
            gradient_recovered=False,
        )

    pr, pp = pearsonr(trust_means, coop_rates)
    sr, sp = spearmanr(trust_means, coop_rates)

    return CrossCulturalResult(
        cluster_results=cluster_results,
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        gradient_recovered=(float(sr) > 0 and float(sp) < 0.10),
    )


def format_cross_cultural_table(result: CrossCulturalResult) -> str:
    """Return a human-readable summary table of cluster results and correlations."""
    lines = [
        "Cross-Cultural Validation Results (Phase 27)",
        "=" * 58,
        f"{'Cluster':<12} {'ESS Trust':>10} {'Sim Coop':>10} {'Gini':>8}",
        "-" * 45,
    ]
    for r in sorted(result.cluster_results, key=lambda x: x.ess_mean_trust):
        lines.append(
            f"{r.cluster_name:<12} "
            f"{r.ess_mean_trust:>10.3f} "
            f"{r.simulated_cooperation_rate:>10.3f} "
            f"{r.simulated_gini:>8.3f}"
        )
    lines += [
        "-" * 45,
        f"Pearson r   = {result.pearson_r:+.3f}  (p = {result.pearson_p:.3f})",
        f"Spearman ρ  = {result.spearman_rho:+.3f}  (p = {result.spearman_p:.3f})",
        f"Gradient recovered: {'YES ✓' if result.gradient_recovered else 'NO ✗'}",
    ]
    return "\n".join(lines)
