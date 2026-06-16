"""Phase 0 — faithful Donor Game substrate (Vallinder & Hughes 2024).

Covers payoff bookkeeping, reputation window, accumulated-resource fitness,
and the DonorGame B_RLHF reference distributions.
"""

from __future__ import annotations

import pytest

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from environment.donor_game import (
    DONATE,
    KEEP,
    DonorDecision,
    DonorGameEngine,
    ReputationLedger,
    always_donate,
    always_keep,
    reputation_threshold,
)
from environment.social_dilemmas import DonorGame, get_game


def _agent(aid: str) -> Agent:
    return Agent(
        profile=AgentProfile(
            agent_id=aid,
            age=35,
            income=1000.0,
            education="secondary",
            occupation="worker",
            location="urban",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
        ),
        state=AgentState(wealth=0.0),
        memory=HierarchicalMemory(max_recent=10),
        policy=MockPolicy(),
    )


def _agents(n: int) -> list[Agent]:
    return [_agent(f"a{i}") for i in range(n)]


# ── Payoff bookkeeping ────────────────────────────────────────────────────────


def test_donate_transfers_cost_and_doubled_benefit():
    eng = DonorGameEngine(_agents(2), always_donate, endowment=10.0, donation=2.0, multiplier=2.0, seed=1)
    eng.run_round(0)
    # Two encounters (each agent donor once). Each donor pays 2, each recipient
    # gets 4. Net per agent: -2 (as donor) + 4 (as recipient) = +2.
    for aid in ("a0", "a1"):
        assert eng.resources[aid] == pytest.approx(12.0)


def test_keep_changes_nothing():
    eng = DonorGameEngine(_agents(4), always_keep, endowment=10.0, donation=2.0, seed=2)
    eng.run_round(0)
    assert all(v == pytest.approx(10.0) for v in eng.resources.values())
    assert eng.history[0].cooperation_rate == 0.0


def test_donation_clamped_to_donor_resources():
    eng = DonorGameEngine(_agents(2), always_donate, endowment=1.0, donation=5.0, multiplier=2.0, seed=3)
    eng.run_round(0)
    # Donor can only give the 1.0 it holds; recipient receives 2.0.
    # Net per agent: -1 + 2 = +1.
    for aid in ("a0", "a1"):
        assert eng.resources[aid] == pytest.approx(2.0)


def test_unaffordable_donation_degrades_to_keep_in_reputation():
    # Donor with zero resources "wants" to donate but cannot → recorded as keep.
    eng = DonorGameEngine(_agents(2), always_donate, endowment=0.0, donation=2.0, seed=4)
    eng.run_round(0)
    assert eng.history[0].cooperation_rate == 0.0
    assert eng.reputation.view("a0").donate_rate == 0.0


# ── Reputation ledger ─────────────────────────────────────────────────────────


def test_reputation_window_evicts_old_actions():
    led = ReputationLedger(window=3)
    for a in (DONATE, KEEP, DONATE, KEEP, KEEP):
        led.record("x", a)
    view = led.view("x")
    assert view.n_observed == 3
    assert view.recent_actions == [DONATE, KEEP, KEEP]
    assert view.donate_rate == pytest.approx(1 / 3)


def test_unknown_recipient_has_empty_reputation():
    led = ReputationLedger()
    view = led.view("never-seen")
    assert view.n_observed == 0
    assert view.donate_rate == 0.0


def test_reputation_threshold_decider_uses_recipient_image():
    # Recipient a1 has a bad image → donor should keep.
    eng = DonorGameEngine(_agents(2), reputation_threshold(0.5), endowment=10.0, donation=2.0, seed=5)
    eng.reputation.record("a1", KEEP)
    eng.reputation.record("a1", KEEP)
    decider = reputation_threshold(0.5)
    decision = decider(None, None, eng.reputation.view("a1"), 0)
    assert decision.action == KEEP


def test_reputation_threshold_donates_to_first_encounter():
    decider = reputation_threshold(0.5)
    led = ReputationLedger()
    decision = decider(None, None, led.view("fresh"), 0)
    assert decision.action == DONATE  # benefit of the doubt


# ── Fitness / ranking ─────────────────────────────────────────────────────────


def test_ranked_agents_orders_by_accumulated_resources():
    eng = DonorGameEngine(_agents(3), always_keep, endowment=10.0, seed=6)
    eng.resources["a0"] = 5.0
    eng.resources["a1"] = 20.0
    eng.resources["a2"] = 12.0
    assert eng.ranked_agents() == ["a1", "a2", "a0"]


def test_action_distribution_sums_to_one():
    eng = DonorGameEngine(_agents(4), reputation_threshold(0.5), endowment=10.0, donation=2.0, seed=7)
    eng.run(num_rounds=12)
    dist = eng.action_distribution()
    assert dist[DONATE] + dist[KEEP] == pytest.approx(1.0)
    assert 0.0 <= dist[DONATE] <= 1.0


# ── Engine guards + decision validation ───────────────────────────────────────


def test_engine_requires_two_agents():
    with pytest.raises(ValueError):
        DonorGameEngine(_agents(1), always_donate)


def test_invalid_decision_action_rejected():
    with pytest.raises(ValueError):
        DonorDecision(action="hoard")


def test_each_agent_donor_and_recipient_once_per_round():
    eng = DonorGameEngine(_agents(6), always_keep, seed=8)
    summary = eng.run_round(0)
    donors = [e.donor_id for e in summary.encounters]
    recipients = [e.recipient_id for e in summary.encounters]
    assert sorted(donors) == [f"a{i}" for i in range(6)]
    assert sorted(recipients) == [f"a{i}" for i in range(6)]


# ── DonorGame social-dilemma references ───────────────────────────────────────


def test_donor_game_registered():
    assert isinstance(get_game("donor_game"), DonorGame)


def test_donor_game_brlhf_overcooperation():
    game = DonorGame()
    # Over-cooperation (RLHF-like) vs human reputation baseline.
    result = game.evaluate({DONATE: 0.95, KEEP: 0.05})
    assert result.cooperation_direction == "over"
    assert game.compute_brlhf({DONATE: 0.95, KEEP: 0.05}, "nash") > 0.0


def test_donor_game_nash_is_keep():
    game = DonorGame()
    assert game.nash_equilibrium == {DONATE: 0.0, KEEP: 1.0}
    assert game.social_optimum == {DONATE: 1.0, KEEP: 0.0}
    assert game.cooperative_actions == [DONATE]
