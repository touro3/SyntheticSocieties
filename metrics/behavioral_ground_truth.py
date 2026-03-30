"""Behavioral ground truth comparison — Gap 1.

Compares BGF simulation outputs against published experimental economics
benchmarks. This closes the critical gap between validating *attitude*
distributions (ESS trust, satisfaction) and validating *behavioral* outputs
(actual decisions: cooperate, work, save).

Without this comparison, BGF's calibration claim is circular: it grounds on
ESS attitudes and validates on ESS attitudes. This module provides the
external behavioral reference point needed to claim that BGF behavioral
outputs are consistent with real human decision-making data.

Published benchmarks
--------------------

Public Goods Games (PGG)
    Ledyard (1995) meta-analysis: 40 experiments, mean contribution rate 0.40–0.60.
    Zelmer (2003) meta-analysis: cooperation rate 0.35–0.55.
    Chaudhuri (2011) review: 40–60% contribution in standard one-shot PGG.
    → BGF benchmark: cooperation_rate ∈ [0.35, 0.55]

Trust Game (Berg et al. 1995)
    Send rate (fraction of endowment sent): mean ≈ 0.50, SD ≈ 0.15.
    Cross-cultural range: 0.20 (some Eastern European samples) – 0.65 (Nordic).
    → BGF benchmark: cooperation_rate ∈ [0.35, 0.65]

Iterated Prisoner's Dilemma (Axelrod 1984 tournament paradigm)
    Cooperation rates: 0.50–0.65 in finite-horizon, 0.60–0.80 in indefinite.
    → BGF benchmark (cooperative agent): cooperation_rate ∈ [0.40, 0.65]

RLHF over-cooperation baseline
    Ungrounded instruction-tuned LLMs (Condition A): cooperation_rate ≈ 0.74.
    This is significantly above all real-human benchmarks — confirming the bias.
    → Expected: ungrounded BGF > all three experimental ranges.

Gini coefficient (wealth inequality)
    ESS-11 European wealth Gini (equivalised disposable income, Eurostat 2023):
        EU-27 average: 0.301
        Nordic: 0.270, Southern: 0.330, Eastern: 0.300
    Income Gini from World Bank (2022): EU countries 0.24–0.38.
    → BGF benchmark (grounded): Gini ∈ [0.20, 0.38]

Usage
-----
>>> from metrics.behavioral_ground_truth import (
...     assess_cooperation_rate,
...     assess_gini,
...     behavioral_ground_truth_report,
... )
>>> report = behavioral_ground_truth_report(
...     simulated_coop_rate=0.38,
...     simulated_gini=0.29,
...     condition="grounded",
... )
>>> print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Published benchmarks ──────────────────────────────────────────────────────

class ExperimentType(str, Enum):
    PUBLIC_GOODS_GAME = "public_goods_game"
    TRUST_GAME = "trust_game"
    ITERATED_PD = "iterated_prisoner_dilemma"
    GINI_WEALTH = "gini_wealth"


@dataclass(frozen=True)
class Benchmark:
    """A published experimental benchmark for a behavioral metric.

    Attributes:
        name: Canonical benchmark name.
        experiment_type: Type of economic experiment.
        metric: The metric being benchmarked ("cooperation_rate" or "gini").
        low: Lower bound of the empirical range.
        high: Upper bound of the empirical range.
        point_estimate: Best single-value estimate (may be None).
        source: Citation for the benchmark.
    """

    name: str
    experiment_type: ExperimentType
    metric: str
    low: float
    high: float
    point_estimate: Optional[float]
    source: str


#: All published behavioral benchmarks used for BGF validation.
BENCHMARKS: list[Benchmark] = [
    Benchmark(
        name="PGG cooperation rate",
        experiment_type=ExperimentType.PUBLIC_GOODS_GAME,
        metric="cooperation_rate",
        low=0.35, high=0.55, point_estimate=0.45,
        source="Ledyard (1995); Zelmer (2003); Chaudhuri (2011)",
    ),
    Benchmark(
        name="Trust game send rate",
        experiment_type=ExperimentType.TRUST_GAME,
        metric="cooperation_rate",
        low=0.35, high=0.65, point_estimate=0.50,
        source="Berg et al. (1995); Johnson & Mislin (2011) meta-analysis",
    ),
    Benchmark(
        name="Iterated PD cooperation",
        experiment_type=ExperimentType.ITERATED_PD,
        metric="cooperation_rate",
        low=0.40, high=0.65, point_estimate=0.53,
        source="Axelrod (1984); Dal Bó & Fréchette (2011)",
    ),
    Benchmark(
        name="EU wealth Gini",
        experiment_type=ExperimentType.GINI_WEALTH,
        metric="gini",
        low=0.20, high=0.38, point_estimate=0.301,
        source="Eurostat ilc_di12 (2023); World Bank (2022)",
    ),
]

#: RLHF ungrounded cooperation rate (BGF Condition A empirical observation).
RLHF_UNGROUNDED_COOP: float = 0.74


# ── Result structures ─────────────────────────────────────────────────────────

class Verdict(str, Enum):
    WITHIN_RANGE = "within_range"
    ABOVE_RANGE  = "above_range"
    BELOW_RANGE  = "below_range"


@dataclass
class BenchmarkComparison:
    """Result of comparing a simulated value against one benchmark.

    Attributes:
        benchmark: The reference benchmark.
        simulated_value: The value from BGF simulation.
        verdict: Whether simulated_value is within, above, or below range.
        deviation: Distance from nearest range boundary (0 if within range).
        standardised_distance: Deviation / (range width / 2); 0 = centre, 1 = edge.
        rlhf_distance: Distance of simulated value from RLHF ungrounded baseline
            (only meaningful for cooperation_rate metric).
    """

    benchmark: Benchmark
    simulated_value: float
    verdict: Verdict
    deviation: float
    standardised_distance: float
    rlhf_distance: Optional[float] = None


@dataclass
class GroundTruthResult:
    """Aggregated behavioral ground truth validation result.

    Attributes:
        condition: "grounded" or "ungrounded".
        simulated_coop_rate: BGF cooperation rate.
        simulated_gini: BGF Gini coefficient.
        coop_comparisons: Per-benchmark comparison for cooperation rate.
        gini_comparisons: Per-benchmark comparison for Gini.
        n_coop_within_range: Number of cooperation benchmarks where value is in range.
        n_gini_within_range: Number of Gini benchmarks where value is in range.
        rlhf_bias_confirmed: True iff ungrounded coop > all experimental ranges.
        grounding_efficacy_confirmed: True iff grounded coop falls within at least one range.
    """

    condition: str
    simulated_coop_rate: float
    simulated_gini: float
    coop_comparisons: list[BenchmarkComparison]
    gini_comparisons: list[BenchmarkComparison]
    n_coop_within_range: int
    n_gini_within_range: int
    rlhf_bias_confirmed: bool
    grounding_efficacy_confirmed: bool


# ── Core functions ────────────────────────────────────────────────────────────

def _compare_to_benchmark(value: float, benchmark: Benchmark) -> BenchmarkComparison:
    """Compare a simulated value against a single benchmark."""
    within = benchmark.low <= value <= benchmark.high

    if within:
        verdict = Verdict.WITHIN_RANGE
        deviation = 0.0
    elif value > benchmark.high:
        verdict = Verdict.ABOVE_RANGE
        deviation = round(value - benchmark.high, 4)
    else:
        verdict = Verdict.BELOW_RANGE
        deviation = round(benchmark.low - value, 4)

    range_half = (benchmark.high - benchmark.low) / 2
    midpoint = (benchmark.low + benchmark.high) / 2
    std_dist = round(abs(value - midpoint) / range_half, 4) if range_half > 0 else 0.0

    rlhf_dist: Optional[float] = None
    if benchmark.metric == "cooperation_rate":
        rlhf_dist = round(RLHF_UNGROUNDED_COOP - value, 4)

    return BenchmarkComparison(
        benchmark=benchmark,
        simulated_value=round(value, 4),
        verdict=verdict,
        deviation=deviation,
        standardised_distance=std_dist,
        rlhf_distance=rlhf_dist,
    )


def assess_cooperation_rate(
    coop_rate: float,
    benchmarks: Optional[list[Benchmark]] = None,
) -> list[BenchmarkComparison]:
    """Compare a simulated cooperation rate against all cooperation benchmarks.

    Args:
        coop_rate: Simulated cooperation rate (fraction of cooperate actions).
        benchmarks: Override default benchmarks (for testing).

    Returns:
        List of BenchmarkComparison, one per cooperation benchmark.
    """
    if benchmarks is None:
        benchmarks = [b for b in BENCHMARKS if b.metric == "cooperation_rate"]
    return [_compare_to_benchmark(coop_rate, b) for b in benchmarks]


def assess_gini(
    gini: float,
    benchmarks: Optional[list[Benchmark]] = None,
) -> list[BenchmarkComparison]:
    """Compare a simulated Gini coefficient against all Gini benchmarks.

    Args:
        gini: Simulated Gini coefficient.
        benchmarks: Override default benchmarks (for testing).

    Returns:
        List of BenchmarkComparison, one per Gini benchmark.
    """
    if benchmarks is None:
        benchmarks = [b for b in BENCHMARKS if b.metric == "gini"]
    return [_compare_to_benchmark(gini, b) for b in benchmarks]


def evaluate(
    simulated_coop_rate: float,
    simulated_gini: float,
    condition: str = "grounded",
) -> GroundTruthResult:
    """Run full behavioral ground truth validation.

    Args:
        simulated_coop_rate: BGF cooperation rate (fraction).
        simulated_gini: BGF Gini coefficient.
        condition: "grounded" (Condition B) or "ungrounded" (Condition A).

    Returns:
        GroundTruthResult with all comparisons and summary flags.
    """
    coop_comps = assess_cooperation_rate(simulated_coop_rate)
    gini_comps = assess_gini(simulated_gini)

    n_coop_within = sum(1 for c in coop_comps if c.verdict == Verdict.WITHIN_RANGE)
    n_gini_within = sum(1 for c in gini_comps if c.verdict == Verdict.WITHIN_RANGE)

    # RLHF bias confirmed: ungrounded rate exceeds all experimental benchmarks
    all_coop_highs = [b.high for b in BENCHMARKS if b.metric == "cooperation_rate"]
    rlhf_bias = condition == "ungrounded" and all(
        simulated_coop_rate > h for h in all_coop_highs
    )

    # Grounding efficacy: grounded rate falls within at least one real-human range
    grounding_ok = condition == "grounded" and n_coop_within >= 1

    return GroundTruthResult(
        condition=condition,
        simulated_coop_rate=round(simulated_coop_rate, 4),
        simulated_gini=round(simulated_gini, 4),
        coop_comparisons=coop_comps,
        gini_comparisons=gini_comps,
        n_coop_within_range=n_coop_within,
        n_gini_within_range=n_gini_within,
        rlhf_bias_confirmed=rlhf_bias,
        grounding_efficacy_confirmed=grounding_ok,
    )


# ── Report ────────────────────────────────────────────────────────────────────

def behavioral_ground_truth_report(
    simulated_coop_rate: float,
    simulated_gini: float,
    condition: str = "grounded",
) -> str:
    """Generate a human-readable behavioral ground truth validation report.

    Args:
        simulated_coop_rate: BGF cooperation rate.
        simulated_gini: BGF Gini coefficient.
        condition: "grounded" or "ungrounded".

    Returns:
        Formatted report string (also printed to stdout).
    """
    result = evaluate(simulated_coop_rate, simulated_gini, condition)

    lines = [
        "=" * 70,
        f"  Behavioral Ground Truth Report  [{condition.upper()}]",
        "=" * 70,
        f"  Simulated cooperation rate: {result.simulated_coop_rate:.4f}",
        f"  RLHF ungrounded baseline:   {RLHF_UNGROUNDED_COOP:.4f}",
        f"  Simulated Gini coefficient: {result.simulated_gini:.4f}",
        "",
    ]

    # ── Cooperation benchmarks ─────────────────────────────────────────────
    lines.append("  Cooperation Rate vs Published Benchmarks:")
    lines.append(f"  {'Benchmark':<30} {'Range':>14} {'Sim':>8} {'Verdict':>14}")
    lines.append("  " + "-" * 68)

    for c in result.coop_comparisons:
        rng = f"[{c.benchmark.low:.2f}, {c.benchmark.high:.2f}]"
        lines.append(
            f"  {c.benchmark.name:<30} {rng:>14} {c.simulated_value:>8.4f}"
            f"  {c.verdict.value:>14}"
        )
    lines.append(
        f"\n  → {result.n_coop_within_range}/{len(result.coop_comparisons)}"
        f" benchmarks within range"
    )

    # ── RLHF bias flag ─────────────────────────────────────────────────────
    if result.rlhf_bias_confirmed:
        lines.append("  → RLHF over-cooperation bias CONFIRMED ✓"
                     " (ungrounded rate > all human benchmarks)")
    elif condition == "grounded" and result.grounding_efficacy_confirmed:
        lines.append("  → Grounding efficacy CONFIRMED ✓"
                     " (grounded rate falls within human behavioral range)")

    # ── Gini benchmarks ───────────────────────────────────────────────────
    lines.append("")
    lines.append("  Gini Coefficient vs Published Benchmarks:")
    lines.append(f"  {'Benchmark':<30} {'Range':>14} {'Sim':>8} {'Verdict':>14}")
    lines.append("  " + "-" * 68)

    for c in result.gini_comparisons:
        rng = f"[{c.benchmark.low:.2f}, {c.benchmark.high:.2f}]"
        lines.append(
            f"  {c.benchmark.name:<30} {rng:>14} {c.simulated_value:>8.4f}"
            f"  {c.verdict.value:>14}"
        )
    lines.append(
        f"\n  → {result.n_gini_within_range}/{len(result.gini_comparisons)}"
        f" Gini benchmarks within range"
    )

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    return report
