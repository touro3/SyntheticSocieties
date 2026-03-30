"""Parameterized policy intervention API for BGF simulations.

Provides a clean interface for injecting institutional policy changes at a
specified simulation round and measuring their causal effect on outcomes.
This is the mechanism needed to test the claim that BGF can serve as
synthetic data for policy evaluation, replacing the need for real experiments.

Supported rules
---------------
redistribute_top_pct
    Tax the top ``parameter`` fraction (e.g. 0.20 = top 20%) of agents by
    wealth, pool the collected tax, and distribute it equally to all agents.
    Models a progressive redistribution / wealth-tax policy.

wealth_floor
    Transfer enough wealth to every agent below ``parameter`` so they reach
    that floor. Funded by proportional levy on agents above the floor.
    Models a universal basic-income / safety-net policy.

cooperation_bonus
    Add ``parameter`` to the wealth delta of every cooperate action this round.
    Models a subsidy or institutional incentive for prosocial behavior.

tax_income
    Levy ``parameter`` fraction (e.g. 0.15 = 15%) on all work-income events
    this round and redistribute equally. Models a flat income tax + dividend.

Usage
-----
>>> from environment.policy_intervention import PolicyIntervention, InterventionEngine
>>> interventions = [
...     PolicyIntervention(trigger_round=10, rule="redistribute_top_pct",
...                        parameter=0.20, label="20%-wealth-tax"),
... ]
>>> engine = InterventionEngine(interventions)
>>> # In simulation loop:
>>> events = engine.apply(round_id=world.state.round_id, agents=agents)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Rule type alias ──────────────────────────────────────────────────────────

RuleName = Literal[
    "redistribute_top_pct",
    "wealth_floor",
    "cooperation_bonus",
    "tax_income",
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PolicyIntervention:
    """A single parameterized policy change applied at a given simulation round.

    Attributes:
        trigger_round: The round at which the intervention fires.
        rule: Which redistribution/incentive rule to apply.
        parameter: The rule-specific numeric parameter (fraction or absolute).
        label: Human-readable label for reporting and plots.
        repeat: If True, the intervention fires every round from trigger_round
            onward. If False (default), it fires exactly once.
    """

    trigger_round: int
    rule: RuleName
    parameter: float
    label: str = ""
    repeat: bool = False

    def __post_init__(self) -> None:
        if self.parameter < 0:
            raise ValueError(f"parameter must be non-negative; got {self.parameter}")
        if self.rule not in {
            "redistribute_top_pct",
            "wealth_floor",
            "cooperation_bonus",
            "tax_income",
        }:
            raise ValueError(f"Unknown rule: '{self.rule}'")
        if not self.label:
            self.label = f"{self.rule}@r{self.trigger_round}(p={self.parameter})"


@dataclass
class InterventionEvent:
    """Record of a single intervention application.

    Attributes:
        round_id: Round in which the intervention fired.
        rule: The rule that was applied.
        parameter: Rule parameter value.
        label: Human-readable label.
        total_transferred: Total wealth transferred across all agents.
        n_agents_affected: Number of agents whose wealth changed.
        agent_deltas: Mapping of agent_id → wealth delta applied.
    """

    round_id: int
    rule: str
    parameter: float
    label: str
    total_transferred: float
    n_agents_affected: int
    agent_deltas: dict[str, float] = field(default_factory=dict)


# ── Rule implementations ─────────────────────────────────────────────────────

def _redistribute_top_pct(agents: list, parameter: float) -> dict[str, float]:
    """Tax the top `parameter` fraction by wealth; distribute pool equally.

    E.g. parameter=0.20 taxes the wealthiest 20% of agents.
    Each taxed agent pays 20% of their wealth above the cut-off.
    Pool is split equally among all agents.
    """
    if not agents:
        return {}
    n = len(agents)
    n_top = max(1, round(n * parameter))

    sorted_agents = sorted(agents, key=lambda a: a.state.wealth, reverse=True)
    top_agents = sorted_agents[:n_top]

    # Each top agent contributes parameter-fraction of their own wealth
    pool = sum(a.state.wealth * parameter for a in top_agents)
    dividend = pool / n

    deltas: dict[str, float] = {}
    for a in top_agents:
        deltas[a.profile.agent_id] = round(-a.state.wealth * parameter + dividend, 6)
    for a in sorted_agents[n_top:]:
        deltas[a.profile.agent_id] = round(dividend, 6)

    return deltas


def _wealth_floor(agents: list, parameter: float) -> dict[str, float]:
    """Guarantee every agent reaches at least `parameter` wealth.

    Shortfall is funded by a proportional levy on agents above the floor.
    If total wealth is insufficient the floor is scaled down proportionally.
    """
    if not agents:
        return {}

    below = [a for a in agents if a.state.wealth < parameter]
    above = [a for a in agents if a.state.wealth >= parameter]

    if not below:
        return {}

    total_shortfall = sum(parameter - a.state.wealth for a in below)
    total_surplus = sum(a.state.wealth - parameter for a in above)

    # Scale floor if system cannot fund it fully
    scale = min(1.0, total_surplus / total_shortfall) if total_shortfall > 0 else 0.0

    deltas: dict[str, float] = {}
    actual_transfer = 0.0
    for a in below:
        delta = round((parameter - a.state.wealth) * scale, 6)
        deltas[a.profile.agent_id] = delta
        actual_transfer += delta

    # Levy proportionally from above-floor agents
    if actual_transfer > 0 and above:
        for a in above:
            share = (a.state.wealth - parameter) / total_surplus if total_surplus > 0 else 0.0
            deltas[a.profile.agent_id] = round(-share * actual_transfer, 6)

    return deltas


def _cooperation_bonus(agents: list, parameter: float) -> dict[str, float]:
    """Add `parameter` wealth to every agent who cooperated this round."""
    deltas: dict[str, float] = {}
    for a in agents:
        if getattr(a.state, "last_action", None) == "cooperate":
            deltas[a.profile.agent_id] = round(parameter, 6)
    return deltas


def _tax_income(agents: list, parameter: float) -> dict[str, float]:
    """Levy `parameter` fraction of work income; redistribute equally.

    Approximates work income as the base work payoff (10.0) × parameter.
    """
    if not agents:
        return {}
    n = len(agents)
    workers = [a for a in agents if getattr(a.state, "last_action", None) == "work"]

    if not workers:
        return {}

    # Estimate tax collected: parameter * work_income per worker
    # We don't deduct from state here — the caller applies `agent_deltas`.
    base_work_income = 10.0  # canonical default from GamePayoffs
    tax_per_worker = base_work_income * parameter
    pool = tax_per_worker * len(workers)
    dividend = pool / n

    deltas: dict[str, float] = {}
    for a in workers:
        deltas[a.profile.agent_id] = round(-tax_per_worker + dividend, 6)
    for a in agents:
        if a not in workers:
            deltas[a.profile.agent_id] = round(dividend, 6)
    return deltas


_RULE_FN = {
    "redistribute_top_pct": _redistribute_top_pct,
    "wealth_floor": _wealth_floor,
    "cooperation_bonus": _cooperation_bonus,
    "tax_income": _tax_income,
}


# ── Engine ───────────────────────────────────────────────────────────────────

class InterventionEngine:
    """Applies a schedule of PolicyInterventions during simulation rounds.

    Pass an instance to ``World.__init__`` or call ``engine.apply()`` from
    the simulation kernel after each round's actions are processed.

    Args:
        interventions: List of PolicyIntervention instances (may be empty).
    """

    def __init__(self, interventions: list[PolicyIntervention] | None = None) -> None:
        self._interventions: list[PolicyIntervention] = interventions or []
        self._fired: set[int] = set()  # indices of single-fire interventions already fired

    def apply(self, round_id: int, agents: list) -> list[InterventionEvent]:
        """Apply all interventions scheduled for `round_id`.

        Modifies agent wealth in-place and returns a list of InterventionEvent
        records (one per intervention that fired).

        Args:
            round_id: Current simulation round.
            agents: List of Agent objects with .state.wealth and .profile.agent_id.

        Returns:
            List of InterventionEvent records (empty if nothing fired).
        """
        events: list[InterventionEvent] = []

        for idx, intervention in enumerate(self._interventions):
            should_fire = (
                round_id == intervention.trigger_round
                or (intervention.repeat and round_id >= intervention.trigger_round)
            )
            already_fired_once = (not intervention.repeat) and (idx in self._fired)

            if not should_fire or already_fired_once:
                continue

            rule_fn = _RULE_FN[intervention.rule]
            deltas = rule_fn(agents, intervention.parameter)

            # Apply deltas to agent states
            for agent in agents:
                delta = deltas.get(agent.profile.agent_id, 0.0)
                if delta:
                    agent.state.wealth = max(0.0, agent.state.wealth + delta)

            total_transferred = sum(abs(d) for d in deltas.values()) / 2
            n_affected = sum(1 for d in deltas.values() if d != 0.0)

            events.append(InterventionEvent(
                round_id=round_id,
                rule=intervention.rule,
                parameter=intervention.parameter,
                label=intervention.label,
                total_transferred=round(total_transferred, 4),
                n_agents_affected=n_affected,
                agent_deltas=deltas,
            ))

            if not intervention.repeat:
                self._fired.add(idx)

        return events

    def has_fired(self, label: str) -> bool:
        """Return True if the intervention with the given label has already fired."""
        for idx, iv in enumerate(self._interventions):
            if iv.label == label and idx in self._fired:
                return True
        return False

    def pending(self, round_id: int) -> list[PolicyIntervention]:
        """Return interventions that have not yet fired and are due at or after round_id."""
        result = []
        for idx, iv in enumerate(self._interventions):
            if iv.repeat:
                if round_id <= iv.trigger_round:
                    result.append(iv)
            elif idx not in self._fired and iv.trigger_round >= round_id:
                result.append(iv)
        return result


# ── Sweep helper ─────────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    """Result of a single policy sweep run.

    Attributes:
        parameter: The policy parameter value tested.
        label: Intervention label.
        pre_gini: Gini coefficient before intervention.
        post_gini: Gini coefficient after last round.
        pre_coop_rate: Cooperation fraction before intervention.
        post_coop_rate: Cooperation fraction after last round.
        delta_gini: post_gini - pre_gini (negative = more equal).
        delta_coop: post_coop_rate - pre_coop_rate.
    """

    parameter: float
    label: str
    pre_gini: float
    post_gini: float
    pre_coop_rate: float
    post_coop_rate: float

    @property
    def delta_gini(self) -> float:
        return round(self.post_gini - self.pre_gini, 6)

    @property
    def delta_coop(self) -> float:
        return round(self.post_coop_rate - self.pre_coop_rate, 6)


def sensitivity_index(sweep_points: list[SweepPoint], outcome: str = "gini") -> float:
    """Estimate ΔOutcome / ΔParameter across a parameter sweep.

    Uses simple linear regression slope as the sensitivity index.
    A negative value for ``gini`` means higher parameter → more equality.

    Args:
        sweep_points: Results from a parameter sweep, ordered by parameter.
        outcome: ``"gini"`` or ``"coop"`` — which delta to use.

    Returns:
        Slope (sensitivity index). Returns 0.0 if fewer than 2 points.
    """
    if len(sweep_points) < 2:
        return 0.0

    xs = [p.parameter for p in sweep_points]
    ys = [p.delta_gini if outcome == "gini" else p.delta_coop for p in sweep_points]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    return round(numerator / denominator, 6) if denominator > 0 else 0.0
