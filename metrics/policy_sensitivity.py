"""Policy sensitivity recovery metrics — Gap 4.

Tests whether BGF correctly recovers the *direction and relative magnitude*
of known cross-cultural and game-theoretic policy effects.

Two complementary validation strategies
----------------------------------------

1. **Direction recovery** (qualitative)
   Given a policy parameter sweep, does the simulation predict the correct
   direction of the effect on Gini / cooperation rate?
   Expectation: redistribution ↑ → Gini ↓; cooperation bonus ↑ → coop rate ↑.

2. **Magnitude calibration** (quantitative)
   Do simulated Gini values across country clusters fall within the empirically
   observed range reported by Eurostat and the World Bank?
   Expectation: simulated Gini ≈ empirical Gini ± tolerance.

Published benchmarks used
--------------------------
Eurostat (2023) Gini coefficients (equivalised disposable income, EU-SILC):
  - Nordic cluster average (NO, SE, DK): 0.27
  - Southern cluster average (IT, ES, PT): 0.33
  - Eastern cluster average (PL, CZ, HU): 0.30
  - EU-27 average: 0.301

Meta-analytic cooperation rates from experimental economics:
  - Public Goods Game (PGG): 0.35–0.55 across 45+ studies (Ledyard 1995;
    Zelmer 2003; Chaudhuri 2011)
  - Trust Game (Berg et al. 1995 paradigm): send rate ≈ 0.50 ± 0.15
  - Prisoner's Dilemma (iterated): cooperation rate ≈ 0.50–0.65

Usage
-----
>>> from metrics.policy_sensitivity import (
...     empirical_gini_benchmarks,
...     gini_magnitude_calibration,
...     direction_recovery,
...     sensitivity_report,
... )
>>> results = [
...     ClusterOutcome("nordic", simulated_gini=0.26, simulated_coop=0.38),
...     ClusterOutcome("southern", simulated_gini=0.35, simulated_coop=0.29),
...     ClusterOutcome("eastern", simulated_gini=0.31, simulated_coop=0.31),
... ]
>>> report = sensitivity_report(results)
>>> print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Published empirical benchmarks ───────────────────────────────────────────

#: Eurostat 2023 Gini coefficients for ESS country clusters (equivalised
#: disposable income). Source: Eurostat ilc_di12 (2023 reference year).
EMPIRICAL_GINI: dict[str, float] = {
    "nordic":   0.27,
    "southern": 0.33,
    "eastern":  0.30,
    "eu_avg":   0.301,
}

#: Empirical cooperation rate range [low, high] from meta-analyses of
#: Public Goods Games (Ledyard 1995; Zelmer 2003; Chaudhuri 2011).
#: Grounded simulation should fall within this range; ungrounded should exceed it.
EMPIRICAL_COOP_RANGE: tuple[float, float] = (0.35, 0.55)

#: Trust game send rate (Berg et al. 1995 paradigm), mean ± 1 SD.
TRUST_GAME_SEND_RATE: tuple[float, float] = (0.35, 0.65)

#: Tolerance for Gini magnitude calibration (absolute deviation from benchmark).
GINI_TOLERANCE: float = 0.06


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ClusterOutcome:
    """Simulated outcome for a single ESS country cluster.

    Attributes:
        cluster_name: One of "nordic", "southern", "eastern".
        simulated_gini: Final-round Gini coefficient from the simulation.
        simulated_coop: Cooperation rate (fraction of cooperate actions).
        condition: "grounded" or "ungrounded" (optional, for labelling).
    """

    cluster_name: str
    simulated_gini: float
    simulated_coop: float
    condition: str = "grounded"


@dataclass
class DirectionResult:
    """Result of a direction-recovery check.

    Attributes:
        check: Description of what was tested.
        expected_direction: The sign expected (+1 = increase, -1 = decrease).
        observed_direction: The sign actually observed.
        recovered: True iff expected_direction == observed_direction.
        delta: The raw difference driving the direction.
        note: Optional explanatory note.
    """

    check: str
    expected_direction: int   # +1 or -1
    observed_direction: int
    recovered: bool
    delta: float
    note: str = ""


@dataclass
class MagnitudeResult:
    """Result of a Gini magnitude calibration check.

    Attributes:
        cluster: Cluster name.
        simulated_gini: Gini from simulation.
        empirical_gini: Published benchmark.
        deviation: abs(simulated - empirical).
        within_tolerance: True iff deviation ≤ GINI_TOLERANCE.
        tolerance: The tolerance used.
    """

    cluster: str
    simulated_gini: float
    empirical_gini: float
    deviation: float
    within_tolerance: bool
    tolerance: float = GINI_TOLERANCE


@dataclass
class CoopCalibrationResult:
    """Result of a cooperation rate calibration check.

    Attributes:
        condition: "grounded" or "ungrounded".
        simulated_coop: Observed cooperation rate.
        benchmark_low: Lower bound of empirical benchmark range.
        benchmark_high: Upper bound of empirical benchmark range.
        within_range: True iff simulated_coop ∈ [benchmark_low, benchmark_high].
        above_range: True iff simulated_coop > benchmark_high (RLHF over-cooperation).
    """

    condition: str
    simulated_coop: float
    benchmark_low: float
    benchmark_high: float
    within_range: bool
    above_range: bool


# ── Core functions ────────────────────────────────────────────────────────────

def empirical_gini_benchmarks() -> dict[str, float]:
    """Return the published Eurostat Gini benchmarks for ESS clusters."""
    return dict(EMPIRICAL_GINI)


def gini_magnitude_calibration(
    cluster_outcomes: list[ClusterOutcome],
    tolerance: float = GINI_TOLERANCE,
) -> list[MagnitudeResult]:
    """Check whether simulated Gini values fall within empirical tolerance.

    Args:
        cluster_outcomes: Simulated results for each cluster.
        tolerance: Maximum acceptable absolute deviation from benchmark.

    Returns:
        List of MagnitudeResult, one per cluster with a known benchmark.
    """
    results = []
    for outcome in cluster_outcomes:
        empirical = EMPIRICAL_GINI.get(outcome.cluster_name)
        if empirical is None:
            continue
        deviation = abs(outcome.simulated_gini - empirical)
        results.append(MagnitudeResult(
            cluster=outcome.cluster_name,
            simulated_gini=round(outcome.simulated_gini, 4),
            empirical_gini=empirical,
            deviation=round(deviation, 4),
            within_tolerance=deviation <= tolerance,
            tolerance=tolerance,
        ))
    return results


def cooperation_calibration(
    outcomes: list[ClusterOutcome],
    benchmark_range: tuple[float, float] = EMPIRICAL_COOP_RANGE,
) -> list[CoopCalibrationResult]:
    """Check whether simulated cooperation rates are calibrated to empirical PGG benchmarks.

    Grounded condition: expected within [0.35, 0.55].
    Ungrounded condition: expected above 0.55 (RLHF over-cooperation).

    Args:
        outcomes: Simulated results, may mix "grounded" and "ungrounded" conditions.
        benchmark_range: (low, high) empirical cooperation rate range.

    Returns:
        List of CoopCalibrationResult, one per outcome.
    """
    low, high = benchmark_range
    results = []
    for o in outcomes:
        within = low <= o.simulated_coop <= high
        above = o.simulated_coop > high
        results.append(CoopCalibrationResult(
            condition=o.condition,
            simulated_coop=round(o.simulated_coop, 4),
            benchmark_low=low,
            benchmark_high=high,
            within_range=within,
            above_range=above,
        ))
    return results


def direction_recovery(
    cluster_outcomes: list[ClusterOutcome],
    policy_parameter_pairs: Optional[list[tuple[float, ClusterOutcome]]] = None,
) -> list[DirectionResult]:
    """Test direction-recovery for cross-cultural and policy sweep signals.

    Two checks are always performed if enough clusters are provided:
      1. Nordic Gini < Eastern Gini (higher-trust clusters should be more equal)
      2. Nordic coop > Eastern coop (higher-trust clusters should cooperate more)

    Optionally, if policy_parameter_pairs is provided (list of (parameter, outcome)),
    the function also checks that higher parameters produce the expected Gini
    reduction (redistribution direction recovery).

    Args:
        cluster_outcomes: Per-cluster simulation results. Must include "nordic"
            and "eastern" for the cross-cultural checks.
        policy_parameter_pairs: Optional list of (parameter, ClusterOutcome) from
            a policy sweep, sorted by increasing parameter.

    Returns:
        List of DirectionResult.
    """
    results: list[DirectionResult] = []
    by_cluster = {o.cluster_name: o for o in cluster_outcomes}

    # Cross-cultural direction checks
    nordic = by_cluster.get("nordic")
    eastern = by_cluster.get("eastern")
    southern = by_cluster.get("southern")

    if nordic and eastern:
        delta_gini = nordic.simulated_gini - eastern.simulated_gini
        results.append(DirectionResult(
            check="nordic_gini < eastern_gini",
            expected_direction=-1,
            observed_direction=-1 if delta_gini < 0 else +1,
            recovered=delta_gini < 0,
            delta=round(delta_gini, 4),
            note="Higher-trust clusters should have lower inequality (Eurostat 2023).",
        ))

        delta_coop = nordic.simulated_coop - eastern.simulated_coop
        results.append(DirectionResult(
            check="nordic_coop > eastern_coop",
            expected_direction=+1,
            observed_direction=+1 if delta_coop > 0 else -1,
            recovered=delta_coop > 0,
            delta=round(delta_coop, 4),
            note="Higher-trust clusters should cooperate more (ESS-11 trust gradient).",
        ))

    if southern and eastern:
        delta_gini = southern.simulated_gini - eastern.simulated_gini
        results.append(DirectionResult(
            check="southern_gini > eastern_gini",
            expected_direction=+1,
            observed_direction=+1 if delta_gini > 0 else -1,
            recovered=delta_gini > 0,
            delta=round(delta_gini, 4),
            note="Southern Europe has higher inequality than Eastern Europe (Eurostat 2023).",
        ))

    # Policy sweep direction check
    if policy_parameter_pairs and len(policy_parameter_pairs) >= 2:
        params = [p for p, _ in policy_parameter_pairs]
        ginis = [o.simulated_gini for _, o in policy_parameter_pairs]
        increasing_param = params[-1] > params[0]
        decreasing_gini = ginis[-1] < ginis[0]

        if increasing_param:
            results.append(DirectionResult(
                check="redistribution_reduces_gini",
                expected_direction=-1,
                observed_direction=-1 if decreasing_gini else +1,
                recovered=decreasing_gini,
                delta=round(ginis[-1] - ginis[0], 4),
                note="Higher redistribution parameter should reduce Gini.",
            ))

    return results


def sensitivity_report(
    cluster_outcomes: list[ClusterOutcome],
    policy_parameter_pairs: Optional[list[tuple[float, ClusterOutcome]]] = None,
    tolerance: float = GINI_TOLERANCE,
) -> str:
    """Generate a human-readable policy sensitivity validation report.

    Args:
        cluster_outcomes: Simulated cluster results.
        policy_parameter_pairs: Optional policy sweep results.
        tolerance: Gini magnitude tolerance.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 65,
        "  Policy Sensitivity Validation Report",
        "=" * 65,
        "",
    ]

    # ── Gini magnitude calibration ─────────────────────────────────────────
    mag_results = gini_magnitude_calibration(cluster_outcomes, tolerance=tolerance)
    if mag_results:
        lines.append("Gini Magnitude Calibration (vs Eurostat 2023):")
        lines.append(f"  {'Cluster':<12} {'Simulated':>10} {'Empirical':>10} {'Deviation':>10} {'Status':>10}")
        lines.append("  " + "-" * 50)
        n_pass = 0
        for r in mag_results:
            status = "PASS ✓" if r.within_tolerance else "FAIL ✗"
            if r.within_tolerance:
                n_pass += 1
            lines.append(
                f"  {r.cluster:<12} {r.simulated_gini:>10.4f} {r.empirical_gini:>10.4f} "
                f"{r.deviation:>10.4f} {status:>10}"
            )
        lines.append(f"  → {n_pass}/{len(mag_results)} clusters within ±{tolerance} tolerance")
        lines.append("")

    # ── Cooperation calibration ────────────────────────────────────────────
    coop_results = cooperation_calibration(cluster_outcomes)
    if coop_results:
        low, high = EMPIRICAL_COOP_RANGE
        lines.append(f"Cooperation Rate Calibration (PGG benchmark: [{low:.2f}, {high:.2f}]):")
        n_within = 0
        n_above = 0
        for r in coop_results:
            status = "WITHIN" if r.within_range else ("ABOVE (RLHF bias)" if r.above_range else "BELOW")
            if r.within_range:
                n_within += 1
            if r.above_range:
                n_above += 1
            lines.append(f"  [{r.condition}]  coop={r.simulated_coop:.4f}  → {status}")
        lines.append(f"  → {n_within}/{len(coop_results)} outcomes within benchmark range")
        if n_above:
            lines.append(f"  → {n_above} outcome(s) above range (RLHF over-cooperation detected)")
        lines.append("")

    # ── Direction recovery ─────────────────────────────────────────────────
    dir_results = direction_recovery(cluster_outcomes, policy_parameter_pairs)
    if dir_results:
        lines.append("Direction Recovery:")
        n_recovered = 0
        for r in dir_results:
            icon = "✓" if r.recovered else "✗"
            n_recovered += int(r.recovered)
            lines.append(f"  {icon} {r.check:<40} Δ={r.delta:+.4f}")
            if r.note:
                lines.append(f"    ({r.note})")
        lines.append(f"  → {n_recovered}/{len(dir_results)} directions correctly recovered")
        lines.append("")

    lines.append("=" * 65)
    report = "\n".join(lines)
    print(report)
    return report


def fraction_gini_within_tolerance(
    cluster_outcomes: list[ClusterOutcome],
    tolerance: float = GINI_TOLERANCE,
) -> float:
    """Return the fraction of clusters whose Gini is within `tolerance` of the benchmark.

    Useful as a single scalar summary for automated test assertions.
    """
    results = gini_magnitude_calibration(cluster_outcomes, tolerance=tolerance)
    if not results:
        return 0.0
    return sum(r.within_tolerance for r in results) / len(results)


def all_directions_recovered(cluster_outcomes: list[ClusterOutcome]) -> bool:
    """Return True iff every direction check passes."""
    results = direction_recovery(cluster_outcomes)
    return bool(results) and all(r.recovered for r in results)
