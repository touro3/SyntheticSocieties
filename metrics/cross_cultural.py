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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr, t as t_dist


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


@dataclass
class ClusterMultiSeedResult:
    """Aggregated multi-seed result for a single ESS country cluster.

    Attributes:
        cluster_name: Cluster key (e.g., "nordic").
        ess_mean_trust: Published ESS-11 cluster mean interpersonal trust [0, 1].
        mean_cooperation_rate: Mean cooperation rate across all seeds.
        std_cooperation_rate: Standard deviation across seeds.
        ci_lower: Lower bound of 95% CI (t-interval).
        ci_upper: Upper bound of 95% CI (t-interval).
        n_seeds: Number of simulation runs (seeds).
        seed_cooperation_rates: Raw per-seed cooperation rates.
        mean_gini: Mean Gini coefficient across seeds.
        n_agents: Agents per run.
        n_rounds: Rounds per run.
    """

    cluster_name: str
    ess_mean_trust: float
    mean_cooperation_rate: float
    std_cooperation_rate: float
    ci_lower: float
    ci_upper: float
    n_seeds: int
    seed_cooperation_rates: list[float] = field(default_factory=list)
    mean_gini: float = 0.0
    n_agents: int = 0
    n_rounds: int = 0
    wvs_trust_pct: Optional[float] = None


def compute_cluster_ci(
    single_results: list[ClusterSimResult],
    confidence: float = 0.95,
) -> ClusterMultiSeedResult:
    """Compute mean ± CI across multiple seeds for a single cluster.

    Args:
        single_results: List of ClusterSimResult for the same cluster, different seeds.
        confidence: Confidence level for the interval (default 0.95).

    Returns:
        ClusterMultiSeedResult with mean, std, and CI.

    Raises:
        ValueError: If single_results is empty or contains multiple cluster names.
    """
    if not single_results:
        raise ValueError("single_results must not be empty.")
    names = {r.cluster_name for r in single_results}
    if len(names) > 1:
        raise ValueError(f"All results must share the same cluster name; got {names}.")

    rates = np.array([r.simulated_cooperation_rate for r in single_results])
    ginis = np.array([r.simulated_gini for r in single_results])
    n = len(rates)

    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates, ddof=1)) if n > 1 else 0.0

    if n > 1:
        se = std_rate / np.sqrt(n)
        margin = float(t_dist.ppf((1 + confidence) / 2, df=n - 1) * se)
    else:
        margin = 0.0

    ref = single_results[0]
    return ClusterMultiSeedResult(
        cluster_name=ref.cluster_name,
        ess_mean_trust=ref.ess_mean_trust,
        mean_cooperation_rate=round(mean_rate, 4),
        std_cooperation_rate=round(std_rate, 4),
        ci_lower=round(max(0.0, mean_rate - margin), 4),
        ci_upper=round(min(1.0, mean_rate + margin), 4),
        n_seeds=n,
        seed_cooperation_rates=[round(float(r), 4) for r in rates],
        mean_gini=round(float(np.mean(ginis)), 4),
        n_agents=ref.n_agents,
        n_rounds=ref.n_rounds,
    )


def compute_cross_cultural_correlation_multiseed(
    multi_results: list[ClusterMultiSeedResult],
) -> CrossCulturalResult:
    """Compute Pearson r and Spearman ρ from multi-seed cluster means.

    Args:
        multi_results: At least 3 ClusterMultiSeedResult instances.

    Returns:
        CrossCulturalResult using mean cooperation rates.
    """
    single = [
        ClusterSimResult(
            cluster_name=r.cluster_name,
            ess_mean_trust=r.ess_mean_trust,
            simulated_cooperation_rate=r.mean_cooperation_rate,
            simulated_gini=r.mean_gini,
            n_agents=r.n_agents,
            n_rounds=r.n_rounds,
        )
        for r in multi_results
    ]
    return compute_cross_cultural_correlation(single)


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
