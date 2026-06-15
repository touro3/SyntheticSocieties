"""Pairwise iterated Donor Game with reputation — substrate for cultural evolution.

Faithful to Vallinder & Hughes (2024, arXiv:2412.10270):

  * Agents are repeatedly paired. In each encounter one agent is the *donor*
    and the other the *recipient*.
  * The donor observes the recipient's **reputation** — a window of the
    recipient's recent donation decisions — and chooses to ``donate`` or
    ``keep``.
  * Donating transfers ``amount`` from the donor; the recipient receives
    ``MULTIPLIER * amount`` (default 2×). Cooperation is individually costly
    but socially optimal.
  * Resources accumulate across rounds; accumulated resources are the fitness
    signal used by the generational selection loop (``simulation.evolution_kernel``).

This module is deliberately decoupled from the world-state ``SimulationKernel``:
the donor game is pairwise and reputation-based, not a shared-resource world.
The B_RLHF reference distributions for the donor game live in
``environment.social_dilemmas.DonorGame``.

The decision interface is a callable ``decision_fn`` so the engine works with
mock/rule-based deciders (tests, dry runs) and the LLM donor policy added in
Phase 1 alike.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

DONATE = "donate"
KEEP = "keep"

MULTIPLIER = 2.0  # recipient benefit = MULTIPLIER * donor cost (V&H default)


# ── Decision + reputation types ───────────────────────────────────────────────


@dataclass
class DonorDecision:
    """A donor's choice in a single encounter."""

    action: str  # DONATE | KEEP
    amount: float | None = None  # donation drawn from donor resources; None → engine default
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.action not in (DONATE, KEEP):
            raise ValueError(f"DonorDecision.action must be {DONATE!r} or {KEEP!r}, got {self.action!r}")


@dataclass
class ReputationView:
    """What a donor is shown about a recipient before deciding."""

    recipient_id: str
    recent_actions: list[str]  # most-recent-last window of the recipient's donor decisions
    donate_rate: float  # fraction of observed actions that were donations
    n_observed: int


class ReputationLedger:
    """Tracks each agent's recent *donor* decisions for reputation lookups.

    Only decisions made while acting as a donor count toward reputation — that
    is the observable behaviour in indirect reciprocity.
    """

    def __init__(self, window: int = 10) -> None:
        if window < 1:
            raise ValueError("reputation window must be >= 1")
        self.window = window
        self._history: dict[str, deque[str]] = {}

    def record(self, agent_id: str, action: str) -> None:
        if action not in (DONATE, KEEP):
            raise ValueError(f"action must be {DONATE!r} or {KEEP!r}, got {action!r}")
        self._history.setdefault(agent_id, deque(maxlen=self.window)).append(action)

    def view(self, recipient_id: str) -> ReputationView:
        hist = list(self._history.get(recipient_id, ()))
        n = len(hist)
        donate_rate = (sum(1 for a in hist if a == DONATE) / n) if n else 0.0
        return ReputationView(
            recipient_id=recipient_id,
            recent_actions=hist,
            donate_rate=donate_rate,
            n_observed=n,
        )


# ── Per-encounter / per-round records ─────────────────────────────────────────


@dataclass
class EncounterRecord:
    round_id: int
    donor_id: str
    recipient_id: str
    action: str
    amount: float  # actual amount transferred (0.0 on keep)
    benefit: float  # amount * MULTIPLIER credited to recipient
    recipient_donate_rate: float  # reputation the donor saw
    reasoning: str = ""


@dataclass
class RoundSummary:
    round_id: int
    n_encounters: int
    cooperation_rate: float  # fraction of donor decisions that were donations
    total_donated: float
    encounters: list[EncounterRecord] = field(default_factory=list)


# A decider sees the donor agent, the recipient agent, the reputation view, and
# the round id, and returns a DonorDecision.
DecisionFn = Callable[["object", "object", ReputationView, int], DonorDecision]


# ── Engine ────────────────────────────────────────────────────────────────────


class DonorGameEngine:
    """Runs pairwise donor-game rounds over a fixed set of agents.

    Resources accumulate in ``self.resources`` (agent_id → float) and are the
    fitness signal for generational selection. Each round every agent acts as a
    donor exactly once and as a recipient exactly once (cyclic random pairing),
    matching V&H's "each agent takes on the roles of donor and recipient".
    """

    def __init__(
        self,
        agents: list,
        decision_fn: DecisionFn,
        *,
        endowment: float = 10.0,
        donation: float = 2.0,
        multiplier: float = MULTIPLIER,
        reputation_window: int = 10,
        seed: int = 0,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DonorGameEngine requires at least 2 agents")
        self.agents = agents
        self.decision_fn = decision_fn
        self.endowment = float(endowment)
        self.donation = float(donation)
        self.multiplier = float(multiplier)
        self.reputation = ReputationLedger(window=reputation_window)
        self.rng = np.random.default_rng(seed)

        self._by_id = {self._aid(a): a for a in agents}
        self.resources: dict[str, float] = {aid: float(endowment) for aid in self._by_id}
        self.history: list[RoundSummary] = []

    @staticmethod
    def _aid(agent) -> str:
        return agent.profile.agent_id

    def _pairings(self) -> list[tuple[str, str]]:
        """Cyclic random pairing: donor k → recipient k+1 (wraps).

        Each agent appears once as donor and once as recipient per round.
        """
        ids = list(self._by_id.keys())
        self.rng.shuffle(ids)
        return [(ids[k], ids[(k + 1) % len(ids)]) for k in range(len(ids))]

    def run_round(self, round_id: int) -> RoundSummary:
        # Resolve the round simultaneously: every encounter decides against the
        # start-of-round resource snapshot and reputation, then all debits and
        # credits are applied together. This makes a round order-independent —
        # a recipient's incoming benefit cannot fund its own donation in the
        # same round, and reputation updates only between rounds (as in V&H).
        encounters: list[EncounterRecord] = []
        n_donate = 0
        total_donated = 0.0
        deltas: dict[str, float] = {aid: 0.0 for aid in self._by_id}

        for donor_id, recipient_id in self._pairings():
            donor = self._by_id[donor_id]
            recipient = self._by_id[recipient_id]
            rep = self.reputation.view(recipient_id)

            decision = self.decision_fn(donor, recipient, rep, round_id)

            if decision.action == DONATE:
                want = self.donation if decision.amount is None else float(decision.amount)
                amount = max(0.0, min(want, self.resources[donor_id]))
            else:
                amount = 0.0

            # A donate decision with zero affordable amount degrades to keep for
            # bookkeeping/reputation purposes (you cannot give what you lack).
            effective_action = DONATE if (decision.action == DONATE and amount > 0.0) else KEEP

            benefit = amount * self.multiplier
            deltas[donor_id] -= amount
            deltas[recipient_id] += benefit

            if effective_action == DONATE:
                n_donate += 1
                total_donated += amount

            encounters.append(
                EncounterRecord(
                    round_id=round_id,
                    donor_id=donor_id,
                    recipient_id=recipient_id,
                    action=effective_action,
                    amount=amount,
                    benefit=benefit,
                    recipient_donate_rate=rep.donate_rate,
                    reasoning=decision.reasoning,
                )
            )

        # Apply accumulated deltas, then record reputation for the round.
        for aid, d in deltas.items():
            self.resources[aid] += d
        for e in encounters:
            self.reputation.record(e.donor_id, e.action)

        coop_rate = n_donate / len(encounters) if encounters else 0.0
        summary = RoundSummary(
            round_id=round_id,
            n_encounters=len(encounters),
            cooperation_rate=coop_rate,
            total_donated=total_donated,
            encounters=encounters,
        )
        self.history.append(summary)
        return summary

    def run(self, num_rounds: int) -> list[RoundSummary]:
        for r in range(num_rounds):
            self.run_round(r)
        return self.history

    # ── Aggregates ────────────────────────────────────────────────────────────

    def cooperation_rate(self) -> float:
        """Donation rate across all encounters run so far."""
        total = sum(s.n_encounters for s in self.history)
        donated = sum(round(s.cooperation_rate * s.n_encounters) for s in self.history)
        return donated / total if total else 0.0

    def action_distribution(self) -> dict[str, float]:
        """Observed {donate, keep} distribution for B_RLHF computation."""
        coop = self.cooperation_rate()
        return {DONATE: coop, KEEP: 1.0 - coop}

    def ranked_agents(self) -> list[str]:
        """Agent ids sorted by accumulated resources, highest first (fitness)."""
        return sorted(self._by_id.keys(), key=lambda aid: self.resources[aid], reverse=True)


# ── Built-in deciders (tests, dry runs, baselines) ────────────────────────────


def always_donate(donor, recipient, rep: ReputationView, round_id: int) -> DonorDecision:
    return DonorDecision(DONATE, reasoning="unconditional cooperator")


def always_keep(donor, recipient, rep: ReputationView, round_id: int) -> DonorDecision:
    return DonorDecision(KEEP, reasoning="unconditional defector")


def reputation_threshold(threshold: float = 0.5) -> DecisionFn:
    """Donate iff the recipient's observed donate-rate ≥ threshold.

    First-encounter recipients (no reputation) are given the benefit of the
    doubt and donated to — the standard "generous" image-scoring strategy.
    """

    def _decide(donor, recipient, rep: ReputationView, round_id: int) -> DonorDecision:
        if rep.n_observed == 0 or rep.donate_rate >= threshold:
            return DonorDecision(DONATE, reasoning=f"recipient image {rep.donate_rate:.2f} ≥ {threshold:.2f}")
        return DonorDecision(KEEP, reasoning=f"recipient image {rep.donate_rate:.2f} < {threshold:.2f}")

    return _decide
