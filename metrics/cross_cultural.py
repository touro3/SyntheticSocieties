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
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as t_dist


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


@dataclass
class HoldoutFitResult:
    """Fit of simulated cooperation against an external trust benchmark.

    Attributes:
        benchmark_name: Either "ess" (in-sample) or "wvs" (out-of-sample holdout).
        pearson_r: Linear association between benchmark trust and simulated cooperation.
        pearson_p: Two-sided p-value for Pearson r.
        spearman_rho: Rank association between benchmark trust and simulated cooperation.
        spearman_p: Two-sided p-value for Spearman ρ.
        gradient_recovered: True iff spearman_rho > 0 and spearman_p < 0.10.
        mae: Mean absolute error between min-max normalised trust and coop vectors.
        rmse: Root mean squared error between min-max normalised trust and coop vectors.
        n_clusters: Number of clusters used in the fit.
    """

    benchmark_name: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    gradient_recovered: bool
    mae: float
    rmse: float
    n_clusters: int


@dataclass
class HoldoutComparison:
    """Condition comparison on a benchmark fit (typically WVS holdout)."""

    benchmark_name: str
    grounded: HoldoutFitResult
    control: HoldoutFitResult
    delta_pearson_r: float
    delta_spearman_rho: float
    delta_mae: float
    delta_rmse: float
    grounded_better: bool


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
        raise ValueError(f"Need at least 3 clusters for meaningful correlation; got {len(cluster_results)}.")

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


def _minmax01(values: np.ndarray) -> np.ndarray:
    """Scale a 1-D vector to [0, 1]; return zeros for constant vectors."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _compute_alignment_error(
    benchmark_values: np.ndarray,
    coop_values: np.ndarray,
) -> tuple[float, float]:
    """Compute scale-free MAE/RMSE between benchmark and cooperation profiles."""
    bx = _minmax01(benchmark_values.astype(float))
    cy = _minmax01(coop_values.astype(float))
    diff = bx - cy
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def compute_benchmark_fit(
    multi_results: list[ClusterMultiSeedResult],
    benchmark: str = "ess",
) -> HoldoutFitResult:
    """Compute fit of simulated cooperation against ESS or WVS trust benchmarks.

    Args:
        multi_results: Per-cluster multi-seed simulation aggregates.
        benchmark: "ess" (in-sample) or "wvs" (out-of-sample holdout).

    Returns:
        HoldoutFitResult with correlation and error metrics.
    """
    if len(multi_results) < 3:
        raise ValueError(f"Need at least 3 clusters for meaningful benchmark fit; got {len(multi_results)}.")

    coop = np.array([r.mean_cooperation_rate for r in multi_results], dtype=float)

    bench_key = benchmark.lower()
    if bench_key == "ess":
        bench = np.array([r.ess_mean_trust for r in multi_results], dtype=float)
    elif bench_key == "wvs":
        missing = [r.cluster_name for r in multi_results if r.wvs_trust_pct is None]
        if missing:
            raise ValueError(
                "WVS benchmark requested but some clusters are missing wvs_trust_pct: " + ", ".join(sorted(missing))
            )
        bench = np.array([float(r.wvs_trust_pct) / 100.0 for r in multi_results], dtype=float)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}. Use 'ess' or 'wvs'.")

    # Degenerate case: no variation in either axis.
    if np.allclose(coop, coop[0]) or np.allclose(bench, bench[0]):
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0
    else:
        pearson_r, pearson_p = pearsonr(bench, coop)
        spearman_rho, spearman_p = spearmanr(bench, coop)
        pearson_r = float(pearson_r)
        pearson_p = float(pearson_p)
        spearman_rho = float(spearman_rho)
        spearman_p = float(spearman_p)

    mae, rmse = _compute_alignment_error(bench, coop)
    return HoldoutFitResult(
        benchmark_name=bench_key,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        gradient_recovered=(spearman_rho > 0 and spearman_p < 0.10),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        n_clusters=len(multi_results),
    )


def compare_holdout_fit(
    grounded_results: list[ClusterMultiSeedResult],
    control_results: list[ClusterMultiSeedResult],
    benchmark: str = "wvs",
) -> HoldoutComparison:
    """Compare grounded vs control condition on a benchmark fit.

    Positive deltas indicate grounded is better:
      - correlation deltas: grounded - control
      - error deltas: control - grounded
    """
    g = compute_benchmark_fit(grounded_results, benchmark=benchmark)
    c = compute_benchmark_fit(control_results, benchmark=benchmark)

    delta_pearson = round(g.pearson_r - c.pearson_r, 4)
    delta_spearman = round(g.spearman_rho - c.spearman_rho, 4)
    delta_mae = round(c.mae - g.mae, 4)
    delta_rmse = round(c.rmse - g.rmse, 4)

    return HoldoutComparison(
        benchmark_name=benchmark.lower(),
        grounded=g,
        control=c,
        delta_pearson_r=delta_pearson,
        delta_spearman_rho=delta_spearman,
        delta_mae=delta_mae,
        delta_rmse=delta_rmse,
        grounded_better=(delta_pearson > 0 and delta_spearman > 0 and delta_rmse > 0),
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


# ── H9: behavioural cross-cultural validation (audit row D.3) ───────────
#
# Pre-registered in docs/construct_validity.md §3 and
# docs/hypothesis_preregistration.md (added 2026-05-13).
#
# H9 tests whether BGF Condition-B country-cluster cooperation rates correlate
# with *behavioural* public-goods-game contribution rates published in:
#
#   Herrmann, Thöni & Gächter (2008). "Antisocial Punishment Across Societies."
#       Science 319(5868), 1362-1367.  Table 1: mean PGG contribution per city.
#   Henrich, Ensminger, McElreath et al. (2010). "Markets, Religion, Community
#       Size, and the Evolution of Fairness and Punishment." Science 327(5972),
#       1480-1484.  Supplementary Table S5: mean DG/UG/TPP offer per site.
#
# Implementation status: skeleton (no GPU). The country-level benchmark
# constants below are *placeholders* derived from the published anchors and
# must be filled by hand-transcription before the H9 paper run. See the
# `H9_TODO` block.

# Hand-transcribed country-mean PGG contribution rates (fraction of endowment
# contributed, round 1; Herrmann 2008 Table 1). Cities aggregated to country
# means where Herrmann reports multiple cities per country.
HERRMANN_PGG_CONTRIBUTION: dict[str, float] = {
    # Western Europe
    "CH": 0.65,  # Zurich / St. Gallen — high
    "DE": 0.64,  # Bonn
    "GB": 0.58,  # Nottingham
    "DK": 0.61,  # Copenhagen
    # Mediterranean
    "GR": 0.34,  # Athens — lowest
    # Eastern Europe
    "RU": 0.50,  # Samara
    "BY": 0.46,  # Minsk
    "UA": 0.40,  # Dnipropetrovsk
    # East Asia (anchor)
    "CN": 0.50,  # Chengdu
    "KR": 0.55,  # Seoul
    # Middle East
    "OM": 0.49,  # Muscat
    "TR": 0.50,  # Istanbul
    # Americas
    "US": 0.55,  # Boston
    # Australia / Oceania
    "AU": 0.55,  # Melbourne
}

# Henrich 2010 dictator-game offers (fraction; Table S5 means, where
# available). Used as the secondary cross-cultural benchmark.
HENRICH_DG_OFFER: dict[str, float] = {
    # WEIRD anchors
    "US": 0.48,
    "GB": 0.43,
    # Small-scale societies (illustrative subset)
    "PE_S": 0.26,  # Sanquianga, Colombia (Henrich Table S5)
    "PG_A": 0.38,  # Au, Papua New Guinea
    "TZ_H": 0.27,  # Hadza, Tanzania
}

# Countries jointly covered by both Herrmann 2008 and an ESS-11 sample.
# Used to size H9's effective n.
H9_JOINT_COUNTRIES = sorted(
    set(HERRMANN_PGG_CONTRIBUTION).intersection(
        {"CH", "DE", "GB", "DK", "GR", "RU", "BY", "UA"}  # ESS-11 sample subset
    )
)


@dataclass
class H9BehavioralResult:
    """Result of correlating BGF cluster cooperation against an external
    behavioural PGG benchmark (Herrmann 2008 or Henrich 2010).

    Attributes:
        benchmark: "herrmann_pgg" or "henrich_dg".
        n_countries: Number of country-level pairs entered into the test.
        spearman_rho: Spearman rank correlation.
        spearman_p: Two-sided p-value (asymptotic; for n < 10 the *exact*
            permutation p-value should also be reported and is computed in
            the runner script).
        pearson_r: Pearson r (sensitivity / descriptive only).
        pearson_p: Two-sided p-value for Pearson r.
        passed: True iff spearman_rho > 0 and spearman_p < 0.05 (H9
            pre-registered threshold).
        per_country: List of (country_code, benchmark_value,
            simulated_cooperation_rate) triples.
    """

    benchmark: str
    n_countries: int
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    passed: bool
    per_country: list[tuple[str, float, float]]


def compute_h9_behavioral_correlation(
    sim_coop_per_country: dict[str, float],
    benchmark: str = "herrmann_pgg",
) -> H9BehavioralResult:
    """H9 cross-cultural behavioural validation (audit D.3).

    Args:
        sim_coop_per_country: ISO country code → BGF Condition-B cooperation
            rate. Caller is responsible for matching country codes to the
            cluster sims (Phase 27 outputs).
        benchmark: One of "herrmann_pgg" (default) or "henrich_dg".

    Returns:
        H9BehavioralResult with Spearman ρ, Pearson r, and a pre-registered
        pass/fail flag.

    Raises:
        ValueError: If the joint country set has fewer than 3 entries
            (Spearman is undefined / has no statistical headroom).
    """
    if benchmark == "herrmann_pgg":
        bench = HERRMANN_PGG_CONTRIBUTION
    elif benchmark == "henrich_dg":
        bench = HENRICH_DG_OFFER
    else:
        raise ValueError(f"Unknown benchmark: {benchmark!r}")

    joint = sorted(set(bench).intersection(sim_coop_per_country))
    if len(joint) < 3:
        raise ValueError(
            f"Need ≥3 joint countries for H9; got {len(joint)} "
            f"(benchmark={benchmark}, sim countries={list(sim_coop_per_country)})."
        )

    bench_vals = np.array([bench[c] for c in joint])
    sim_vals = np.array([sim_coop_per_country[c] for c in joint])

    # Degenerate-cooperation guard mirrors compute_cross_cultural_correlation.
    if np.all(sim_vals == sim_vals[0]) or np.all(bench_vals == bench_vals[0]):
        return H9BehavioralResult(
            benchmark=benchmark,
            n_countries=len(joint),
            spearman_rho=0.0,
            spearman_p=1.0,
            pearson_r=0.0,
            pearson_p=1.0,
            passed=False,
            per_country=[(c, float(bench[c]), float(sim_coop_per_country[c])) for c in joint],
        )

    sr, sp = spearmanr(bench_vals, sim_vals)
    pr, pp = pearsonr(bench_vals, sim_vals)

    return H9BehavioralResult(
        benchmark=benchmark,
        n_countries=len(joint),
        spearman_rho=float(sr),
        spearman_p=float(sp),
        pearson_r=float(pr),
        pearson_p=float(pp),
        passed=(float(sr) > 0 and float(sp) < 0.05),
        per_country=[(c, float(bench[c]), float(sim_coop_per_country[c])) for c in joint],
    )


# ── H9_TODO ───────────────────────────────────────────────────────────────
# Before treating H9 results as the official paper number:
#   1. Replace the placeholder constants above with values transcribed
#      directly from Herrmann 2008 Table 1 and Henrich 2010 Table S5; cite
#      page numbers in commit messages.
#   2. Run the LLM-scale Condition-B simulation for each country in
#      H9_JOINT_COUNTRIES via scripts/run_cross_cultural.py (already
#      Phase 27 infrastructure); emit per-country cooperation rates.
#   3. Persist H9BehavioralResult to analysis/tables/h9_cross_cultural_behavioral.json
#      so analysis/forest_plot.py can pick the row up automatically.
# Until (1)-(3) land, H9 is reported as ⏳ pending in evidence_audit.md row D.3
# and rendered as a pending row in the forest plot.
