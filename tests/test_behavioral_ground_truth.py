"""Tests for metrics/behavioral_ground_truth.py."""

from __future__ import annotations

from metrics.behavioral_ground_truth import (
    BENCHMARKS,
    RLHF_UNGROUNDED_COOP,
    Benchmark,
    ExperimentType,
    Verdict,
    assess_cooperation_rate,
    assess_gini,
    behavioral_ground_truth_report,
    evaluate,
)

# ── BENCHMARKS sanity checks ──────────────────────────────────────────────────


def test_benchmarks_list_nonempty():
    assert len(BENCHMARKS) >= 4


def test_all_benchmarks_have_valid_ranges():
    for b in BENCHMARKS:
        assert b.low >= 0.0
        assert b.high <= 1.0
        assert b.low < b.high, f"Invalid range for {b.name}: [{b.low}, {b.high}]"


def test_rlhf_ungrounded_above_all_coop_benchmarks():
    """The RLHF ungrounded rate should exceed every experimental upper bound."""
    coop_benchmarks = [b for b in BENCHMARKS if b.metric == "cooperation_rate"]
    for b in coop_benchmarks:
        assert RLHF_UNGROUNDED_COOP > b.high, (
            f"RLHF rate {RLHF_UNGROUNDED_COOP} not above {b.name} upper bound {b.high}"
        )


def test_benchmarks_cover_both_metrics():
    metrics = {b.metric for b in BENCHMARKS}
    assert "cooperation_rate" in metrics
    assert "gini" in metrics


def test_benchmark_sources_nonempty():
    for b in BENCHMARKS:
        assert b.source, f"Benchmark {b.name} has no source citation"


# ── assess_cooperation_rate ───────────────────────────────────────────────────


def test_grounded_coop_within_pgg_range():
    # 0.42 is inside PGG [0.35, 0.55]
    comps = assess_cooperation_rate(0.42)
    pgg = next(c for c in comps if c.benchmark.experiment_type == ExperimentType.PUBLIC_GOODS_GAME)
    assert pgg.verdict == Verdict.WITHIN_RANGE


def test_ungrounded_coop_above_all_ranges():
    comps = assess_cooperation_rate(RLHF_UNGROUNDED_COOP)
    for c in comps:
        assert c.verdict == Verdict.ABOVE_RANGE, (
            f"{c.benchmark.name}: expected ABOVE_RANGE for coop={RLHF_UNGROUNDED_COOP}"
        )


def test_very_low_coop_below_range():
    comps = assess_cooperation_rate(0.05)
    for c in comps:
        assert c.verdict == Verdict.BELOW_RANGE


def test_coop_boundary_low_inclusive():
    # Exactly at lower bound of PGG range (0.35)
    comps = assess_cooperation_rate(0.35)
    pgg = next(c for c in comps if c.benchmark.experiment_type == ExperimentType.PUBLIC_GOODS_GAME)
    assert pgg.verdict == Verdict.WITHIN_RANGE


def test_coop_boundary_high_inclusive():
    comps = assess_cooperation_rate(0.55)
    pgg = next(c for c in comps if c.benchmark.experiment_type == ExperimentType.PUBLIC_GOODS_GAME)
    assert pgg.verdict == Verdict.WITHIN_RANGE


def test_coop_deviation_zero_when_within():
    comps = assess_cooperation_rate(0.45)
    for c in comps:
        if c.verdict == Verdict.WITHIN_RANGE:
            assert c.deviation == 0.0


def test_coop_deviation_positive_when_above():
    comps = assess_cooperation_rate(0.74)
    for c in comps:
        assert c.deviation > 0.0
        assert c.verdict == Verdict.ABOVE_RANGE


def test_coop_rlhf_distance_computed():
    comps = assess_cooperation_rate(0.40)
    for c in comps:
        expected = round(RLHF_UNGROUNDED_COOP - 0.40, 4)
        assert abs(c.rlhf_distance - expected) < 1e-4


def test_coop_standardised_distance_at_midpoint():
    # Midpoint of PGG [0.35, 0.55] = 0.45 → standardised distance = 0
    comps = assess_cooperation_rate(0.45)
    pgg = next(c for c in comps if c.benchmark.experiment_type == ExperimentType.PUBLIC_GOODS_GAME)
    assert pgg.standardised_distance < 0.01


def test_custom_benchmarks_respected():
    custom = [
        Benchmark(
            name="custom",
            experiment_type=ExperimentType.TRUST_GAME,
            metric="cooperation_rate",
            low=0.60,
            high=0.80,
            point_estimate=0.70,
            source="test",
        )
    ]
    comps = assess_cooperation_rate(0.50, benchmarks=custom)
    assert len(comps) == 1
    assert comps[0].verdict == Verdict.BELOW_RANGE


# ── assess_gini ───────────────────────────────────────────────────────────────


def test_gini_within_eu_range():
    comps = assess_gini(0.29)
    assert all(c.verdict == Verdict.WITHIN_RANGE for c in comps)


def test_gini_above_range():
    comps = assess_gini(0.55)
    assert all(c.verdict == Verdict.ABOVE_RANGE for c in comps)


def test_gini_below_range():
    comps = assess_gini(0.05)
    assert all(c.verdict == Verdict.BELOW_RANGE for c in comps)


def test_gini_rlhf_distance_is_none():
    comps = assess_gini(0.30)
    for c in comps:
        assert c.rlhf_distance is None


# ── evaluate ──────────────────────────────────────────────────────────────────


def test_evaluate_grounded_within_range():
    result = evaluate(simulated_coop_rate=0.42, simulated_gini=0.29, condition="grounded")
    assert result.grounding_efficacy_confirmed
    assert not result.rlhf_bias_confirmed
    assert result.n_coop_within_range >= 1


def test_evaluate_ungrounded_confirms_rlhf_bias():
    result = evaluate(simulated_coop_rate=0.74, simulated_gini=0.18, condition="ungrounded")
    assert result.rlhf_bias_confirmed
    assert not result.grounding_efficacy_confirmed


def test_evaluate_grounded_not_flagged_as_rlhf():
    result = evaluate(0.42, 0.29, condition="grounded")
    assert not result.rlhf_bias_confirmed


def test_evaluate_ungrounded_low_coop_no_rlhf_flag():
    # Ungrounded but coop is within range — bias not confirmed
    result = evaluate(0.50, 0.30, condition="ungrounded")
    assert not result.rlhf_bias_confirmed


def test_evaluate_returns_correct_simulated_values():
    result = evaluate(0.38, 0.31, condition="grounded")
    assert result.simulated_coop_rate == 0.38
    assert result.simulated_gini == 0.31


def test_evaluate_comparisons_nonempty():
    result = evaluate(0.40, 0.30, condition="grounded")
    assert len(result.coop_comparisons) > 0
    assert len(result.gini_comparisons) > 0


def test_evaluate_grounded_coop_low_not_efficacy_confirmed():
    # Grounded but cooperation rate is below all ranges
    result = evaluate(simulated_coop_rate=0.05, simulated_gini=0.29, condition="grounded")
    assert not result.grounding_efficacy_confirmed


# ── behavioral_ground_truth_report ────────────────────────────────────────────


def test_report_returns_string():
    report = behavioral_ground_truth_report(0.42, 0.29, condition="grounded")
    assert isinstance(report, str)


def test_report_contains_key_sections():
    report = behavioral_ground_truth_report(0.42, 0.29, condition="grounded")
    assert "Cooperation Rate" in report
    assert "Gini" in report
    assert "GROUNDED" in report


def test_report_grounding_confirmed_message():
    report = behavioral_ground_truth_report(0.42, 0.29, condition="grounded")
    assert "Grounding efficacy CONFIRMED" in report


def test_report_rlhf_confirmed_message():
    report = behavioral_ground_truth_report(0.74, 0.18, condition="ungrounded")
    assert "RLHF over-cooperation bias CONFIRMED" in report


def test_report_within_range_shown():
    report = behavioral_ground_truth_report(0.45, 0.29, condition="grounded")
    assert "within_range" in report


def test_report_above_range_for_ungrounded():
    report = behavioral_ground_truth_report(0.74, 0.18, condition="ungrounded")
    assert "above_range" in report
