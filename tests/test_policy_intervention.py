"""Tests for metrics/policy_intervention.py.8."""

from __future__ import annotations

import pytest

from metrics.policy_intervention import (
    InterventionResult,
    InterventionSummary,
    _gini,
    _hash_uniform,
    aggregate_by_intensity,
    run_intervention_sweep,
    run_single_intervention,
)

# ── Unit tests ────────────────────────────────────────────────────────────


class TestHelpers:
    def test_hash_uniform_in_range(self):
        v = _hash_uniform("agent_0001", 5)
        assert 0.0 <= v < 1.0

    def test_hash_uniform_deterministic(self):
        a = _hash_uniform("agent_0001", 5)
        b = _hash_uniform("agent_0001", 5)
        assert a == b

    def test_hash_uniform_different_inputs(self):
        a = _hash_uniform("agent_0001", 5)
        b = _hash_uniform("agent_0001", 6)
        c = _hash_uniform("agent_0002", 5)
        assert a != b
        assert a != c

    def test_gini_uniform(self):
        g = _gini([10.0, 10.0, 10.0, 10.0])
        assert abs(g) < 1e-10

    def test_gini_max_inequality(self):
        values = [0.0, 0.0, 0.0, 100.0]
        g = _gini(values)
        assert g > 0.5

    def test_gini_empty(self):
        assert _gini([]) == 0.0

    def test_gini_zero_sum(self):
        assert _gini([0.0, 0.0, 0.0]) == 0.0


class TestRunSingleIntervention:
    def test_returns_correct_type(self):
        result = run_single_intervention(intensity=0.0, n_agents=20, n_rounds=10, seed=42)
        assert isinstance(result, InterventionResult)

    def test_per_round_length(self):
        result = run_single_intervention(intensity=0.0, n_agents=20, n_rounds=15, seed=42)
        assert len(result.per_round_cooperation) == 15
        assert len(result.per_round_wealth_mean) == 15

    def test_cooperation_rates_in_range(self):
        result = run_single_intervention(intensity=0.0, n_agents=50, n_rounds=10, seed=42)
        for v in result.per_round_cooperation:
            assert 0.0 <= v <= 1.0

    def test_gini_in_range(self):
        result = run_single_intervention(intensity=0.0, n_agents=50, n_rounds=10, seed=42)
        assert 0.0 <= result.gini_final <= 1.0

    def test_wealth_positive(self):
        result = run_single_intervention(intensity=0.0, n_agents=50, n_rounds=10, seed=42)
        assert result.wealth_mean_final > 0.0

    def test_delta_cooperation_sign_with_boost(self):
        no_boost = run_single_intervention(intensity=0.0, n_agents=100, n_rounds=30, seed=42, intervention_round=15)
        boosted = run_single_intervention(intensity=0.20, n_agents=100, n_rounds=30, seed=42, intervention_round=15)
        assert boosted.delta_cooperation > no_boost.delta_cooperation

    def test_zero_intensity_delta_near_zero(self):
        result = run_single_intervention(intensity=0.0, n_agents=100, n_rounds=30, seed=42, intervention_round=15)
        assert abs(result.delta_cooperation) < 0.15

    def test_fields_consistent(self):
        result = run_single_intervention(intensity=0.10, n_agents=50, n_rounds=20, seed=7)
        assert result.intensity == 0.10
        assert result.n_agents == 50
        assert result.n_rounds == 20
        assert result.seed == 7
        expected_delta = result.cooperation_rate_post - result.cooperation_rate_pre
        assert abs(result.delta_cooperation - expected_delta) < 1e-9

    def test_reproducibility_across_calls(self):
        r1 = run_single_intervention(intensity=0.05, n_agents=30, n_rounds=10, seed=99)
        r2 = run_single_intervention(intensity=0.05, n_agents=30, n_rounds=10, seed=99)
        assert r1.delta_cooperation == r2.delta_cooperation
        assert r1.gini_final == r2.gini_final

    def test_different_seeds_differ(self):
        r1 = run_single_intervention(intensity=0.10, n_agents=50, n_rounds=15, seed=1)
        r2 = run_single_intervention(intensity=0.10, n_agents=50, n_rounds=15, seed=2)
        assert r1.gini_final != r2.gini_final

    def test_intervention_round_zero(self):
        result = run_single_intervention(intensity=0.10, n_agents=30, n_rounds=10, seed=42, intervention_round=0)
        assert isinstance(result, InterventionResult)
        assert len(result.per_round_cooperation) == 10


class TestRunInterventionSweep:
    def test_sweep_length(self):
        results = run_intervention_sweep(
            intensities=(0.0, 0.10),
            seeds=(42, 123),
            n_agents=20,
            n_rounds=5,
        )
        assert len(results) == 4

    def test_all_intensities_covered(self):
        intensities = (0.0, 0.05, 0.10, 0.20)
        results = run_intervention_sweep(
            intensities=intensities,
            seeds=(42,),
            n_agents=20,
            n_rounds=5,
        )
        found = {r.intensity for r in results}
        assert found == set(intensities)

    def test_all_seeds_covered(self):
        results = run_intervention_sweep(
            intensities=(0.0,),
            seeds=(1, 2, 3),
            n_agents=20,
            n_rounds=5,
        )
        found = {r.seed for r in results}
        assert found == {1, 2, 3}


class TestAggregateByIntensity:
    def test_aggregation_count(self):
        results = run_intervention_sweep(
            intensities=(0.0, 0.10, 0.20),
            seeds=(42, 123),
            n_agents=20,
            n_rounds=5,
        )
        summaries = aggregate_by_intensity(results)
        assert len(summaries) == 3

    def test_summaries_sorted_ascending(self):
        results = run_intervention_sweep(
            intensities=(0.20, 0.0, 0.10),
            seeds=(42,),
            n_agents=20,
            n_rounds=5,
        )
        summaries = aggregate_by_intensity(results)
        intensities = [s.intensity for s in summaries]
        assert intensities == sorted(intensities)

    def test_summary_type(self):
        results = run_intervention_sweep(intensities=(0.0,), seeds=(42,), n_agents=20, n_rounds=5)
        summaries = aggregate_by_intensity(results)
        assert isinstance(summaries[0], InterventionSummary)

    def test_intensity_pct_format(self):
        results = run_intervention_sweep(
            intensities=(0.0, 0.05, 0.10, 0.20),
            seeds=(42,),
            n_agents=20,
            n_rounds=5,
        )
        summaries = aggregate_by_intensity(results)
        pcts = [s.intensity_pct for s in summaries]
        assert "0%" in pcts
        assert "5%" in pcts
        assert "10%" in pcts
        assert "20%" in pcts

    def test_delta_coop_monotone_with_intensity(self):
        results = run_intervention_sweep(
            intensities=(0.0, 0.05, 0.10, 0.20),
            seeds=(42, 123, 7),
            n_agents=100,
            n_rounds=30,
            intervention_round=15,
        )
        summaries = aggregate_by_intensity(results)
        deltas = [s.delta_coop_mean for s in summaries]
        for i in range(1, len(deltas)):
            assert deltas[i] >= deltas[i - 1] - 0.01

    def test_per_round_length_matches_n_rounds(self):
        results = run_intervention_sweep(intensities=(0.0,), seeds=(42,), n_agents=20, n_rounds=15)
        summaries = aggregate_by_intensity(results)
        assert len(summaries[0].per_round_cooperation) == 15


from environment.policy_intervention import (
    InterventionEngine,
    PolicyIntervention,
    SweepPoint,
    sensitivity_index,
)
from tests.conftest import make_agent

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_agents(wealths: list[float]) -> list:
    agents = []
    for i, w in enumerate(wealths):
        a = make_agent(agent_id=f"agent_{i}", wealth=w)
        agents.append(a)
    return agents


def total_wealth(agents: list) -> float:
    return sum(a.state.wealth for a in agents)


# ── PolicyIntervention validation ─────────────────────────────────────────────


def test_invalid_negative_parameter():
    with pytest.raises(ValueError, match="non-negative"):
        PolicyIntervention(trigger_round=5, rule="wealth_floor", parameter=-1.0)


def test_invalid_rule():
    with pytest.raises(ValueError, match="Unknown rule"):
        PolicyIntervention(trigger_round=5, rule="magic_money", parameter=0.1)


def test_auto_label_generated():
    iv = PolicyIntervention(trigger_round=3, rule="tax_income", parameter=0.15)
    assert "tax_income" in iv.label
    assert "0.15" in iv.label


def test_custom_label_preserved():
    iv = PolicyIntervention(trigger_round=3, rule="tax_income", parameter=0.15, label="my_label")
    assert iv.label == "my_label"


# ── redistribute_top_pct ──────────────────────────────────────────────────────


def test_redistribute_top_pct_conserves_wealth():
    """Total wealth should be conserved (±floating-point tolerance)."""
    agents = make_agents([200.0, 150.0, 100.0, 50.0, 10.0])
    before = total_wealth(agents)

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="redistribute_top_pct", parameter=0.20)])
    engine.apply(round_id=1, agents=agents)

    after = total_wealth(agents)
    assert abs(after - before) < 1e-4, f"Wealth not conserved: {before:.4f} → {after:.4f}"


def test_redistribute_top_pct_reduces_gini():
    """After redistribution the richest agent should be less wealthy relative to others."""
    agents = make_agents([500.0, 100.0, 100.0, 100.0, 100.0])
    richest_before = max(a.state.wealth for a in agents)

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="redistribute_top_pct", parameter=0.20)])
    engine.apply(round_id=1, agents=agents)

    richest_after = max(a.state.wealth for a in agents)
    poorest_after = min(a.state.wealth for a in agents)
    assert richest_after < richest_before
    assert poorest_after > 100.0  # poorest gained


def test_redistribute_does_not_fire_at_wrong_round():
    agents = make_agents([200.0, 50.0])
    before = [a.state.wealth for a in agents]

    engine = InterventionEngine([PolicyIntervention(trigger_round=5, rule="redistribute_top_pct", parameter=0.20)])
    engine.apply(round_id=3, agents=agents)

    for i, a in enumerate(agents):
        assert a.state.wealth == before[i]


# ── wealth_floor ─────────────────────────────────────────────────────────────


def test_wealth_floor_lifts_poor():
    agents = make_agents([200.0, 150.0, 10.0, 5.0])
    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="wealth_floor", parameter=50.0)])
    engine.apply(round_id=1, agents=agents)

    for a in agents:
        assert a.state.wealth >= 50.0 or abs(a.state.wealth - 50.0) < 1e-3


def test_wealth_floor_no_effect_if_all_above():
    agents = make_agents([100.0, 200.0, 300.0])
    before = [a.state.wealth for a in agents]

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="wealth_floor", parameter=50.0)])
    events = engine.apply(round_id=1, agents=agents)

    assert events[0].n_agents_affected == 0
    for i, a in enumerate(agents):
        assert a.state.wealth == before[i]


def test_wealth_floor_conserves_wealth():
    agents = make_agents([300.0, 200.0, 20.0, 5.0])
    before = total_wealth(agents)

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="wealth_floor", parameter=80.0)])
    engine.apply(round_id=1, agents=agents)

    after = total_wealth(agents)
    assert abs(after - before) < 1e-3


# ── cooperation_bonus ─────────────────────────────────────────────────────────


def test_cooperation_bonus_only_rewards_cooperators():
    agents = make_agents([100.0, 100.0, 100.0])
    agents[0].state.last_action = "cooperate"
    agents[1].state.last_action = "work"
    agents[2].state.last_action = "save"

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="cooperation_bonus", parameter=5.0)])
    engine.apply(round_id=1, agents=agents)

    assert agents[0].state.wealth == 105.0
    assert agents[1].state.wealth == 100.0
    assert agents[2].state.wealth == 100.0


def test_cooperation_bonus_no_cooperators_no_effect():
    agents = make_agents([100.0, 100.0])
    for a in agents:
        a.state.last_action = "work"

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="cooperation_bonus", parameter=10.0)])
    events = engine.apply(round_id=1, agents=agents)

    assert events[0].n_agents_affected == 0
    assert all(a.state.wealth == 100.0 for a in agents)


# ── tax_income ───────────────────────────────────────────────────────────────


def test_tax_income_workers_pay_non_workers_receive():
    agents = make_agents([100.0, 100.0, 100.0])
    agents[0].state.last_action = "work"
    agents[1].state.last_action = "work"
    agents[2].state.last_action = "save"

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="tax_income", parameter=0.30)])
    engine.apply(round_id=1, agents=agents)

    # Workers should lose net (paid tax > received dividend share)
    # Non-worker should gain
    assert agents[2].state.wealth > 100.0


def test_tax_income_no_workers_no_effect():
    agents = make_agents([100.0, 100.0])
    for a in agents:
        a.state.last_action = "save"

    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="tax_income", parameter=0.15)])
    events = engine.apply(round_id=1, agents=agents)

    assert events[0].n_agents_affected == 0


# ── Single-fire behaviour ─────────────────────────────────────────────────────


def test_single_fire_does_not_repeat():
    agents = make_agents([200.0, 50.0])
    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="redistribute_top_pct", parameter=0.20)])

    events_r1 = engine.apply(round_id=1, agents=agents)
    wealth_after_r1 = [a.state.wealth for a in agents]

    events_r2 = engine.apply(round_id=1, agents=agents)  # same round again
    wealth_after_r2 = [a.state.wealth for a in agents]

    assert len(events_r1) == 1
    assert len(events_r2) == 0  # already fired, should not repeat
    assert wealth_after_r1 == wealth_after_r2


def test_repeat_fires_every_round():
    agents = make_agents([200.0, 50.0])
    engine = InterventionEngine(
        [PolicyIntervention(trigger_round=1, rule="cooperation_bonus", parameter=0.0, repeat=True)]
    )

    e1 = engine.apply(round_id=1, agents=agents)
    e2 = engine.apply(round_id=2, agents=agents)
    e3 = engine.apply(round_id=3, agents=agents)

    assert len(e1) == 1
    assert len(e2) == 1
    assert len(e3) == 1


# ── InterventionEngine.has_fired / pending ────────────────────────────────────


def test_has_fired_after_apply():
    agents = make_agents([100.0, 50.0])
    engine = InterventionEngine(
        [PolicyIntervention(trigger_round=2, rule="wealth_floor", parameter=30.0, label="floor_30")]
    )

    assert not engine.has_fired("floor_30")
    engine.apply(round_id=2, agents=agents)
    assert engine.has_fired("floor_30")


def test_pending_lists_unfired():
    engine = InterventionEngine(
        [
            PolicyIntervention(trigger_round=5, rule="tax_income", parameter=0.10, label="tax_r5"),
            PolicyIntervention(trigger_round=10, rule="wealth_floor", parameter=20.0, label="floor_r10"),
        ]
    )

    pending = engine.pending(round_id=1)
    assert len(pending) == 2


# ── InterventionEvent content ─────────────────────────────────────────────────


def test_event_records_total_transferred():
    agents = make_agents([200.0, 100.0, 50.0])
    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="redistribute_top_pct", parameter=0.20)])
    events = engine.apply(round_id=1, agents=agents)

    assert len(events) == 1
    ev = events[0]
    assert ev.round_id == 1
    assert ev.rule == "redistribute_top_pct"
    assert ev.total_transferred > 0
    assert ev.n_agents_affected > 0


def test_empty_engine_returns_no_events():
    agents = make_agents([100.0])
    engine = InterventionEngine()
    events = engine.apply(round_id=5, agents=agents)
    assert events == []


def test_no_agent_below_zero():
    """Wealth must never go negative after an intervention."""
    agents = make_agents([1.0, 0.5, 0.1])
    engine = InterventionEngine([PolicyIntervention(trigger_round=1, rule="redistribute_top_pct", parameter=0.99)])
    engine.apply(round_id=1, agents=agents)

    for a in agents:
        assert a.state.wealth >= 0.0


# ── sensitivity_index ─────────────────────────────────────────────────────────


def test_sensitivity_index_negative_for_redistribution():
    """Higher redistribution parameter should produce more negative Gini delta."""
    points = [
        SweepPoint(0.10, "p0.10", pre_gini=0.4, post_gini=0.38, pre_coop_rate=0.3, post_coop_rate=0.3),
        SweepPoint(0.20, "p0.20", pre_gini=0.4, post_gini=0.35, pre_coop_rate=0.3, post_coop_rate=0.3),
        SweepPoint(0.30, "p0.30", pre_gini=0.4, post_gini=0.32, pre_coop_rate=0.3, post_coop_rate=0.3),
    ]
    idx = sensitivity_index(points, outcome="gini")
    assert idx < 0, f"Expected negative sensitivity for redistribution; got {idx}"


def test_sensitivity_index_zero_with_one_point():
    points = [
        SweepPoint(0.10, "p0.10", pre_gini=0.4, post_gini=0.38, pre_coop_rate=0.3, post_coop_rate=0.3),
    ]
    assert sensitivity_index(points) == 0.0


def test_sensitivity_index_coop_outcome():
    points = [
        SweepPoint(5.0, "bonus_5", pre_gini=0.3, post_gini=0.3, pre_coop_rate=0.30, post_coop_rate=0.35),
        SweepPoint(10.0, "bonus_10", pre_gini=0.3, post_gini=0.3, pre_coop_rate=0.30, post_coop_rate=0.42),
    ]
    idx = sensitivity_index(points, outcome="coop")
    assert idx > 0, f"Larger bonus should increase cooperation; sensitivity={idx}"


def test_sweep_point_delta_properties():
    sp = SweepPoint(
        parameter=0.20,
        label="test",
        pre_gini=0.40,
        post_gini=0.35,
        pre_coop_rate=0.30,
        post_coop_rate=0.38,
    )
    assert abs(sp.delta_gini - (-0.05)) < 1e-6
    assert abs(sp.delta_coop - 0.08) < 1e-6
