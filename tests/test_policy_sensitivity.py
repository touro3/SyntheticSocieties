"""Tests for metrics/policy_sensitivity.py."""

from __future__ import annotations

from metrics.policy_sensitivity import (
    EMPIRICAL_COOP_RANGE,
    EMPIRICAL_GINI,
    ClusterOutcome,
    all_directions_recovered,
    cooperation_calibration,
    direction_recovery,
    empirical_gini_benchmarks,
    fraction_gini_within_tolerance,
    gini_magnitude_calibration,
    sensitivity_report,
)

# ── empirical_gini_benchmarks ─────────────────────────────────────────────────


def test_benchmarks_contain_known_clusters():
    benchmarks = empirical_gini_benchmarks()
    assert "nordic" in benchmarks
    assert "southern" in benchmarks
    assert "eastern" in benchmarks
    assert "eu_avg" in benchmarks


def test_benchmarks_are_plausible():
    benchmarks = empirical_gini_benchmarks()
    for name, val in benchmarks.items():
        assert 0.0 < val < 1.0, f"Gini for {name} out of [0,1]: {val}"
    # Nordic should be most equal
    assert benchmarks["nordic"] < benchmarks["southern"]
    assert benchmarks["nordic"] < benchmarks["eastern"]


def test_benchmarks_returns_copy():
    b1 = empirical_gini_benchmarks()
    b1["nordic"] = 999.0
    b2 = empirical_gini_benchmarks()
    assert b2["nordic"] != 999.0


# ── gini_magnitude_calibration ────────────────────────────────────────────────


def _realistic_outcomes() -> list[ClusterOutcome]:
    """Simulated Gini values close to Eurostat benchmarks."""
    return [
        ClusterOutcome("nordic", simulated_gini=0.27, simulated_coop=0.40),
        ClusterOutcome("southern", simulated_gini=0.34, simulated_coop=0.31),
        ClusterOutcome("eastern", simulated_gini=0.31, simulated_coop=0.33),
    ]


def test_calibration_pass_for_realistic_outcomes():
    results = gini_magnitude_calibration(_realistic_outcomes())
    assert all(r.within_tolerance for r in results), [
        f"{r.cluster}: Δ={r.deviation}" for r in results if not r.within_tolerance
    ]


def test_calibration_fail_for_unrealistic_gini():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.55, simulated_coop=0.40),
    ]
    results = gini_magnitude_calibration(outcomes)
    assert len(results) == 1
    assert not results[0].within_tolerance


def test_calibration_skips_unknown_cluster():
    outcomes = [ClusterOutcome("martian", simulated_gini=0.30, simulated_coop=0.40)]
    results = gini_magnitude_calibration(outcomes)
    assert results == []


def test_calibration_deviation_is_absolute():
    outcomes = [ClusterOutcome("nordic", simulated_gini=0.22, simulated_coop=0.40)]
    results = gini_magnitude_calibration(outcomes)
    expected_dev = abs(0.22 - EMPIRICAL_GINI["nordic"])
    assert abs(results[0].deviation - expected_dev) < 1e-6


def test_custom_tolerance_respected():
    outcomes = [ClusterOutcome("nordic", simulated_gini=0.30, simulated_coop=0.40)]
    # 0.30 vs 0.27 → deviation 0.03
    r_tight = gini_magnitude_calibration(outcomes, tolerance=0.02)
    r_loose = gini_magnitude_calibration(outcomes, tolerance=0.05)
    assert not r_tight[0].within_tolerance
    assert r_loose[0].within_tolerance


# ── fraction_gini_within_tolerance ───────────────────────────────────────────


def test_fraction_all_within():
    frac = fraction_gini_within_tolerance(_realistic_outcomes())
    assert frac == 1.0


def test_fraction_none_within():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.60, simulated_coop=0.40),
        ClusterOutcome("eastern", simulated_gini=0.65, simulated_coop=0.33),
    ]
    frac = fraction_gini_within_tolerance(outcomes)
    assert frac == 0.0


def test_fraction_empty_returns_zero():
    assert fraction_gini_within_tolerance([]) == 0.0


# ── cooperation_calibration ───────────────────────────────────────────────────


def test_grounded_coop_within_range():
    outcomes = [ClusterOutcome("nordic", simulated_gini=0.27, simulated_coop=0.42, condition="grounded")]
    results = cooperation_calibration(outcomes)
    assert results[0].within_range


def test_ungrounded_coop_above_range():
    outcomes = [ClusterOutcome("full_pop", simulated_gini=0.20, simulated_coop=0.74, condition="ungrounded")]
    results = cooperation_calibration(outcomes)
    assert results[0].above_range
    assert not results[0].within_range


def test_coop_below_range():
    outcomes = [ClusterOutcome("cluster", simulated_gini=0.30, simulated_coop=0.10, condition="grounded")]
    results = cooperation_calibration(outcomes)
    assert not results[0].within_range
    assert not results[0].above_range


def test_coop_calibration_boundary_inclusive():
    low, high = EMPIRICAL_COOP_RANGE
    outcomes = [
        ClusterOutcome("a", simulated_gini=0.30, simulated_coop=low),
        ClusterOutcome("b", simulated_gini=0.30, simulated_coop=high),
    ]
    results = cooperation_calibration(outcomes)
    assert results[0].within_range
    assert results[1].within_range


# ── direction_recovery ────────────────────────────────────────────────────────


def test_direction_recovery_all_correct():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.26, simulated_coop=0.42),
        ClusterOutcome("southern", simulated_gini=0.34, simulated_coop=0.30),
        ClusterOutcome("eastern", simulated_gini=0.30, simulated_coop=0.33),
    ]
    results = direction_recovery(outcomes)
    assert len(results) >= 3
    for r in results:
        assert r.recovered, f"Direction not recovered: {r.check} (Δ={r.delta})"


def test_direction_recovery_nordic_eastern_only():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.25, simulated_coop=0.45),
        ClusterOutcome("eastern", simulated_gini=0.31, simulated_coop=0.31),
    ]
    results = direction_recovery(outcomes)
    checks = {r.check for r in results}
    assert "nordic_gini < eastern_gini" in checks
    assert "nordic_coop > eastern_coop" in checks


def test_direction_recovery_failure_detected():
    # Reversed Gini: nordic HIGHER than eastern (wrong direction)
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.40, simulated_coop=0.45),
        ClusterOutcome("eastern", simulated_gini=0.28, simulated_coop=0.31),
    ]
    results = direction_recovery(outcomes)
    gini_check = next(r for r in results if "gini" in r.check and "nordic" in r.check)
    assert not gini_check.recovered


def test_policy_sweep_direction_recovery():
    pairs = [
        (0.10, ClusterOutcome("test", simulated_gini=0.39, simulated_coop=0.30)),
        (0.20, ClusterOutcome("test", simulated_gini=0.36, simulated_coop=0.31)),
        (0.30, ClusterOutcome("test", simulated_gini=0.33, simulated_coop=0.32)),
    ]
    results = direction_recovery([], policy_parameter_pairs=pairs)
    sweep_check = next(r for r in results if "redistribution" in r.check)
    assert sweep_check.recovered


def test_policy_sweep_wrong_direction():
    # Gini increases as parameter increases — wrong direction
    pairs = [
        (0.10, ClusterOutcome("test", simulated_gini=0.30, simulated_coop=0.30)),
        (0.30, ClusterOutcome("test", simulated_gini=0.40, simulated_coop=0.30)),
    ]
    results = direction_recovery([], policy_parameter_pairs=pairs)
    sweep_check = next((r for r in results if "redistribution" in r.check), None)
    if sweep_check:
        assert not sweep_check.recovered


def test_direction_recovery_missing_clusters_no_error():
    # Only southern cluster provided — cross-cultural checks should be skipped
    outcomes = [ClusterOutcome("southern", simulated_gini=0.33, simulated_coop=0.29)]
    results = direction_recovery(outcomes)
    checks = {r.check for r in results}
    assert "nordic_gini < eastern_gini" not in checks


# ── all_directions_recovered ──────────────────────────────────────────────────


def test_all_directions_recovered_true():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.25, simulated_coop=0.44),
        ClusterOutcome("southern", simulated_gini=0.34, simulated_coop=0.29),
        ClusterOutcome("eastern", simulated_gini=0.30, simulated_coop=0.33),
    ]
    assert all_directions_recovered(outcomes)


def test_all_directions_recovered_false_when_one_fails():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.40, simulated_coop=0.44),  # wrong Gini
        ClusterOutcome("eastern", simulated_gini=0.28, simulated_coop=0.33),
    ]
    assert not all_directions_recovered(outcomes)


def test_all_directions_recovered_empty():
    # No nordic/eastern → no direction checks → returns False (no checks run)
    assert not all_directions_recovered([])


# ── sensitivity_report ────────────────────────────────────────────────────────


def test_sensitivity_report_returns_string(capsys):
    outcomes = _realistic_outcomes()
    report = sensitivity_report(outcomes)
    assert isinstance(report, str)
    assert "Gini" in report
    assert "Cooperation" in report
    assert "Direction" in report


def test_sensitivity_report_shows_pass():
    report = sensitivity_report(_realistic_outcomes())
    assert "PASS" in report


def test_sensitivity_report_shows_fail_for_bad_gini():
    outcomes = [
        ClusterOutcome("nordic", simulated_gini=0.55, simulated_coop=0.40),
        ClusterOutcome("southern", simulated_gini=0.60, simulated_coop=0.31),
        ClusterOutcome("eastern", simulated_gini=0.58, simulated_coop=0.33),
    ]
    report = sensitivity_report(outcomes)
    assert "FAIL" in report


def test_sensitivity_report_with_policy_sweep():
    outcomes = _realistic_outcomes()
    pairs = [
        (0.10, ClusterOutcome("test", simulated_gini=0.38, simulated_coop=0.30)),
        (0.25, ClusterOutcome("test", simulated_gini=0.33, simulated_coop=0.32)),
    ]
    report = sensitivity_report(outcomes, policy_parameter_pairs=pairs)
    assert "redistribution" in report or "Direction" in report
