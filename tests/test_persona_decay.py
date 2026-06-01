"""Tests for persona decay metrics.

Limitations and failure mode analysis.
"""

from __future__ import annotations

from metrics.persona_decay import (
    compute_decay_summary,
    compute_per_round_persona_fidelity,
    expected_cooperation_rate,
)
from tests.conftest import make_agent, make_profile

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_event(agent_id: str, round_id: int, action: str) -> dict:
    """Build a minimal event dict matching events.jsonl format."""
    return {
        "agent_id": agent_id,
        "round_id": round_id,
        "action": {"action_type": action, "amount": 5.0},
    }


def _make_events_for_agent(agent_id: str, rounds: int, action_sequence: list[str]) -> list[dict]:
    """Build events for an agent cycling through action_sequence."""
    return [_make_event(agent_id, r, action_sequence[r % len(action_sequence)]) for r in range(1, rounds + 1)]


# ── expected_cooperation_rate ────────────────────────────────────────────
#
# NOTE on expected values: the empirical model is fitted on Austrian ESS
# volunteering data (base rate 18.0%). All predictions therefore cluster
# near 10-25%. This is the scientifically correct expectation — the prior
# heuristic (0.2 + 0.6 * trust * (1 - risk)) was inflating cooperation
# expectations to 20-80% without empirical support.
#
# Key empirical finding from the fitted model (see data/cooperation_model.json):
#   - risk_tolerance is a SIGNIFICANT POSITIVE predictor (higher risk → more
#     volunteering, CIs exclude zero). This contradicts the heuristic assumption
#     that risk aversion promotes cooperation.
#   - trust_people is NOT a significant predictor (95% CI includes zero).
#   - social_activity is a significant positive predictor.


class TestExpectedCooperationRate:
    def test_high_risk_predicts_more_cooperation_than_low_risk(self):
        """Empirical finding: risk tolerance is a positive predictor of volunteering.

        The heuristic assumed the opposite (higher risk → less cooperation).
        The fitted logistic regression on ESS data shows risk_taking is a
        significant positive predictor (95% bootstrap CI excludes zero).
        """
        profile_high_risk = make_profile(trust_people=0.5, risk_tolerance=0.9)
        profile_low_risk = make_profile(trust_people=0.5, risk_tolerance=0.1)
        rate_high = expected_cooperation_rate(profile_high_risk)
        rate_low = expected_cooperation_rate(profile_low_risk)
        assert rate_high > rate_low, (
            f"Expected high_risk({rate_high:.3f}) > low_risk({rate_low:.3f}). "
            "This reflects the empirical finding that risk tolerance positively "
            "predicts volunteering in ESS Round 11 (Austria)."
        )

    def test_high_social_activity_predicts_more_cooperation(self):
        """social_activity is a significant positive predictor (CI excludes zero)."""
        profile_active = make_profile(trust_people=0.5, risk_tolerance=0.5, social_activity=0.9)
        profile_passive = make_profile(trust_people=0.5, risk_tolerance=0.5, social_activity=0.1)
        rate_active = expected_cooperation_rate(profile_active)
        rate_passive = expected_cooperation_rate(profile_passive)
        assert rate_active > rate_passive

    def test_predictions_near_empirical_base_rate(self):
        """All predictions should be within a realistic range around the 18% base rate.

        The fitted model is bounded by its data. Extreme predictions far from
        the 18% volunteering rate signal a model or imputation error.
        """
        for trust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for risk in [0.0, 0.25, 0.5, 0.75, 1.0]:
                profile = make_profile(trust_people=trust, risk_tolerance=risk)
                rate = expected_cooperation_rate(profile)
                assert 0.0 <= rate <= 1.0, f"trust={trust}, risk={risk} => {rate}"
                # Sanity: ESS base rate is 18%; even extremes should stay in 5-45% range
                assert 0.05 <= rate <= 0.45, (
                    f"Prediction {rate:.3f} is implausibly far from the 18% base rate "
                    f"(trust={trust}, risk={risk}). Check model coefficients."
                )

    def test_none_trust_defaults_gracefully(self):
        profile = make_profile(trust_people=None, risk_tolerance=0.5)
        rate = expected_cooperation_rate(profile)
        assert 0.0 <= rate <= 1.0

    def test_none_risk_defaults_gracefully(self):
        profile = make_profile(trust_people=0.5, risk_tolerance=None)
        rate = expected_cooperation_rate(profile)
        assert 0.0 <= rate <= 1.0

    def test_none_all_attributes_returns_reasonable_default(self):
        """Profiles with no ESS attributes should predict near the population mean."""
        profile = make_profile(trust_people=None, risk_tolerance=None, social_activity=None)
        rate = expected_cooperation_rate(profile)
        # Should be near 18% base rate (all features imputed to training means)
        assert 0.05 <= rate <= 0.40


# ── compute_per_round_persona_fidelity ───────────────────────────────────


class TestPerRoundPersonaFidelity:
    def test_empty_events_returns_empty(self):
        profile = make_profile(trust_people=0.8, risk_tolerance=0.2)
        result = compute_per_round_persona_fidelity([], profile)
        assert result["rounds"] == []
        assert result["fidelity"] == []
        assert result["decay_rate"] == 0.0

    def test_single_round(self):
        profile = make_profile(trust_people=0.9, risk_tolerance=0.1)
        events = [_make_event("agent_0", 1, "cooperate")]
        result = compute_per_round_persona_fidelity(events, profile, window=1)
        assert len(result["rounds"]) == 1
        assert len(result["fidelity"]) == 1
        assert 0.0 <= result["fidelity"][0] <= 1.0

    def test_agent_matching_empirical_rate_has_higher_fidelity(self):
        """An agent cooperating at the empirically predicted rate has higher fidelity
        than one that cooperates at 100%.

        The empirical model predicts ~18% cooperation (Austrian base rate).
        An agent cooperating 100% of the time is far from that expectation and
        should have lower fidelity than one cooperating at the predicted rate.
        """
        profile = make_profile(trust_people=0.5, risk_tolerance=0.5)
        from metrics.persona_decay import expected_cooperation_rate as ecr

        expected = ecr(profile)  # ~18-20%

        # Agent cooperating approximately at the empirical rate (every 5th action)
        # Use a cycle of [cooperate, work, work, work, work] → 20% coop rate
        events_on_rate = _make_events_for_agent("agent_0", 20, ["cooperate", "work", "work", "work", "work"])
        # Agent cooperating 100% — far above the empirically predicted rate
        events_all_coop = _make_events_for_agent("agent_0", 20, ["cooperate"])

        result_on_rate = compute_per_round_persona_fidelity(events_on_rate, profile, window=5)
        result_all_coop = compute_per_round_persona_fidelity(events_all_coop, profile, window=5)

        avg_fidelity_on_rate = sum(result_on_rate["fidelity"]) / len(result_on_rate["fidelity"])
        avg_fidelity_all_coop = sum(result_all_coop["fidelity"]) / len(result_all_coop["fidelity"])

        assert avg_fidelity_on_rate > avg_fidelity_all_coop, (
            f"Agent at empirical rate ({avg_fidelity_on_rate:.3f}) should have "
            f"higher fidelity than always-cooperate ({avg_fidelity_all_coop:.3f}). "
            f"Expected cooperation rate = {expected:.3f}."
        )

    def test_contradictory_agent_has_lower_fidelity_than_matching(self):
        """An agent cooperating never has lower fidelity than one cooperating at
        the empirical rate (both are deviations, but never < sometimes)."""
        profile = make_profile(trust_people=0.5, risk_tolerance=0.5)
        # Agent cooperating sometimes (closer to ~18% empirical rate)
        events_sometimes = _make_events_for_agent("agent_0", 20, ["cooperate", "work", "work", "work", "work"])
        # Agent never cooperating (further from ~18% than sometimes cooperating)
        events_never = _make_events_for_agent("agent_0", 20, ["work"])

        result_sometimes = compute_per_round_persona_fidelity(events_sometimes, profile, window=5)
        result_never = compute_per_round_persona_fidelity(events_never, profile, window=5)

        avg_sometimes = sum(result_sometimes["fidelity"]) / len(result_sometimes["fidelity"])
        avg_never = sum(result_never["fidelity"]) / len(result_never["fidelity"])

        assert avg_sometimes > avg_never

    def test_output_structure(self):
        profile = make_profile(trust_people=0.5, risk_tolerance=0.5)
        events = _make_events_for_agent("agent_0", 10, ["work", "save", "cooperate"])
        result = compute_per_round_persona_fidelity(events, profile, window=3)
        assert "rounds" in result
        assert "fidelity" in result
        assert "decay_rate" in result
        assert "half_life" in result
        assert isinstance(result["decay_rate"], float)

    def test_decay_rate_negative_for_drifting_agent(self):
        """An agent that starts matching its persona then drifts should have a
        negative decay rate (fidelity decreasing over time).

        With the empirical model (base rate ~18%), an agent must start near
        the empirically-predicted cooperation rate and then deviate. We use
        a profile whose expected rate is ~20% (high risk, mean social_activity)
        and an agent that initially cooperates every 5th round (~20%), then
        drops to never cooperating.
        """
        profile = make_profile(trust_people=0.5, risk_tolerance=0.9, social_activity=0.5)
        # First 15 rounds: cooperate every 5th action (~20%), close to expected ~20-23%
        on_rate_actions = ["cooperate", "work", "work", "work", "work"]
        events_on_rate = _make_events_for_agent("agent_0", 15, on_rate_actions)
        # Rounds 16-30: never cooperate (drift to 0%)
        events_drift = [_make_event("agent_0", r, "work") for r in range(16, 31)]

        result = compute_per_round_persona_fidelity(events_on_rate + events_drift, profile, window=5)
        # With drift from ~20% to 0%, fidelity should trend downward
        assert result["decay_rate"] <= 0.0, (
            f"Decay rate {result['decay_rate']:.4f} should be ≤ 0 for an agent "
            "that matches its persona early then drifts to no cooperation."
        )


# ── compute_decay_summary ────────────────────────────────────────────────


class TestDecaySummary:
    def test_empty_events_returns_structure(self):
        agents = [make_agent("agent_0", trust_people=0.5, risk_tolerance=0.5)]
        result = compute_decay_summary([], agents)
        assert "per_agent" in result
        assert "mean_fidelity_per_round" in result
        assert "mean_decay_rate" in result
        assert "agents_drifted_pct" in result

    def test_multiple_agents(self):
        agents = [
            make_agent("agent_0", trust_people=0.9, risk_tolerance=0.1),
            make_agent("agent_1", trust_people=0.1, risk_tolerance=0.9),
        ]
        events = _make_events_for_agent("agent_0", 10, ["cooperate"]) + _make_events_for_agent("agent_1", 10, ["work"])
        # Fix agent_ids in events
        for e in events[10:]:
            e["agent_id"] = "agent_1"

        result = compute_decay_summary(events, agents, window=3)
        assert "agent_0" in result["per_agent"]
        assert "agent_1" in result["per_agent"]
        assert isinstance(result["agents_drifted_pct"], float)
        assert 0.0 <= result["agents_drifted_pct"] <= 100.0

    def test_mean_fidelity_per_round_has_values(self):
        agents = [make_agent("agent_0", trust_people=0.5, risk_tolerance=0.5)]
        events = _make_events_for_agent("agent_0", 10, ["work", "cooperate"])
        result = compute_decay_summary(events, agents, window=3)
        assert len(result["mean_fidelity_per_round"]) > 0
