"""Tests for persona decay metrics.

Phase 24 — Limitations and failure mode analysis.
"""

from __future__ import annotations

import pytest

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


def _make_events_for_agent(
    agent_id: str, rounds: int, action_sequence: list[str]
) -> list[dict]:
    """Build events for an agent cycling through action_sequence."""
    return [
        _make_event(agent_id, r, action_sequence[r % len(action_sequence)])
        for r in range(1, rounds + 1)
    ]


# ── expected_cooperation_rate ────────────────────────────────────────────


class TestExpectedCooperationRate:
    def test_high_trust_low_risk_cooperates_more(self):
        profile = make_profile(trust_people=0.9, risk_tolerance=0.1)
        rate = expected_cooperation_rate(profile)
        assert rate > 0.6

    def test_low_trust_high_risk_cooperates_less(self):
        profile = make_profile(trust_people=0.1, risk_tolerance=0.9)
        rate = expected_cooperation_rate(profile)
        assert rate < 0.3

    def test_bounded_zero_one(self):
        for trust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for risk in [0.0, 0.25, 0.5, 0.75, 1.0]:
                profile = make_profile(trust_people=trust, risk_tolerance=risk)
                rate = expected_cooperation_rate(profile)
                assert 0.0 <= rate <= 1.0, f"trust={trust}, risk={risk} => {rate}"

    def test_none_trust_defaults_gracefully(self):
        profile = make_profile(trust_people=None, risk_tolerance=0.5)
        rate = expected_cooperation_rate(profile)
        assert 0.0 <= rate <= 1.0

    def test_none_risk_defaults_gracefully(self):
        profile = make_profile(trust_people=0.5, risk_tolerance=None)
        rate = expected_cooperation_rate(profile)
        assert 0.0 <= rate <= 1.0


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

    def test_consistent_agent_maintains_high_fidelity(self):
        """An agent whose behavior matches its persona should have high fidelity."""
        profile = make_profile(trust_people=0.9, risk_tolerance=0.1)
        # High trust, low risk -> expected to cooperate often
        events = _make_events_for_agent("agent_0", 20, ["cooperate"])
        result = compute_per_round_persona_fidelity(events, profile, window=5)
        # Should have high fidelity throughout
        for f in result["fidelity"]:
            assert f > 0.5

    def test_contradictory_agent_has_low_fidelity(self):
        """An agent whose behavior contradicts its persona should have low fidelity."""
        profile = make_profile(trust_people=0.9, risk_tolerance=0.1)
        # High trust should cooperate, but we make it only work
        events = _make_events_for_agent("agent_0", 20, ["work"])
        result = compute_per_round_persona_fidelity(events, profile, window=5)
        # Should have low fidelity
        for f in result["fidelity"]:
            assert f < 0.8

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
        """An agent that starts cooperative then shifts should have negative decay rate."""
        profile = make_profile(trust_people=0.9, risk_tolerance=0.1)
        # Start cooperating, then switch to only working
        events = (
            _make_events_for_agent("agent_0", 10, ["cooperate"])
            + [_make_event("agent_0", r, "work") for r in range(11, 21)]
        )
        result = compute_per_round_persona_fidelity(events, profile, window=3)
        # decay_rate should be negative (fidelity decreasing)
        assert result["decay_rate"] <= 0.0


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
        events = (
            _make_events_for_agent("agent_0", 10, ["cooperate"])
            + _make_events_for_agent("agent_1", 10, ["work"])
        )
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
