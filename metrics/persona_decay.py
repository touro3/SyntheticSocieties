"""Persona decay metrics — quantifying behavioral drift from initial persona.

Phase 24 — Limitations and failure mode analysis.

Measures how well an agent's behavior over time remains consistent with
the cooperation expectations set by its ESS-derived persona attributes.

Key concept:
  expected_cooperation_rate(profile) maps trust and risk attributes to a
  baseline cooperation probability. This is an explicit, documented
  assumption -- intentionally simple (linear) so it can be critiqued and
  replaced.

  Persona fidelity at round r = 1 - |actual_coop_rate - expected_coop_rate|
  Decay rate = linear slope of fidelity over rounds (negative = drifting)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

# ── Expected cooperation rate ────────────────────────────────────────────

_DEFAULT_TRUST = 0.5
_DEFAULT_RISK = 0.5


def expected_cooperation_rate(profile: Any) -> float:
    """Map ESS trust/risk attributes to an expected cooperation baseline.

    Model:  expected_coop = 0.2 + 0.6 * trust_people * (1 - risk_tolerance)

    This yields:
      - High trust, low risk  -> ~0.74 (cooperate often)
      - Low trust, high risk  -> ~0.26 (cooperate rarely)
      - Neutral (0.5, 0.5)    -> ~0.35

    The 0.2 floor ensures all agents have some baseline cooperation
    probability, and 0.8 ceiling prevents deterministic cooperation.

    Args:
        profile: AgentProfile (or any object with trust_people and
            risk_tolerance attributes).

    Returns:
        Float in [0.2, 0.8].
    """
    trust = getattr(profile, "trust_people", None)
    risk = getattr(profile, "risk_tolerance", None)

    if trust is None:
        trust = _DEFAULT_TRUST
    if risk is None:
        risk = _DEFAULT_RISK

    return 0.2 + 0.6 * trust * (1.0 - risk)


# ── Per-round persona fidelity ───────────────────────────────────────────


def compute_per_round_persona_fidelity(
    events: list[dict],
    profile: Any,
    window: int = 5,
) -> dict[str, Any]:
    """Compute persona fidelity at each round for a single agent.

    For each round r with at least ``window`` events ending at r,
    compute the actual cooperation rate in [r-window+1, r] and compare
    to the expected rate from the agent's profile.

    Fidelity = 1 - |actual_coop_rate - expected_coop_rate|

    Args:
        events: Event dicts for a **single agent** (pre-filtered).
        profile: AgentProfile with trust_people and risk_tolerance.
        window: Sliding window size in rounds.

    Returns:
        {
            'rounds': list[int],
            'fidelity': list[float],
            'decay_rate': float,     # Linear slope (negative = decaying)
            'half_life': int | None, # Round where fidelity drops below 0.5
        }
    """
    if not events:
        return {"rounds": [], "fidelity": [], "decay_rate": 0.0, "half_life": None}

    agent_id = profile.agent_id
    expected_coop = expected_cooperation_rate(profile)

    # Group actions by round for this agent
    round_actions: dict[int, list[str]] = defaultdict(list)
    for e in events:
        if e.get("agent_id") != agent_id:
            continue
        rid = e.get("round_id")
        action = e.get("action", {}).get("action_type")
        if rid is not None and action is not None:
            round_actions[rid].append(action)

    if not round_actions:
        return {"rounds": [], "fidelity": [], "decay_rate": 0.0, "half_life": None}

    sorted_rounds = sorted(round_actions.keys())
    rounds_out: list[int] = []
    fidelity_out: list[float] = []

    for i, r in enumerate(sorted_rounds):
        # Collect actions in the window ending at r
        window_start_idx = max(0, i - window + 1)
        window_rounds = sorted_rounds[window_start_idx : i + 1]

        all_actions = []
        for wr in window_rounds:
            all_actions.extend(round_actions[wr])

        if not all_actions:
            continue

        actual_coop = sum(1 for a in all_actions if a == "cooperate") / len(all_actions)
        fidelity = 1.0 - abs(actual_coop - expected_coop)
        fidelity = max(0.0, min(1.0, fidelity))

        rounds_out.append(r)
        fidelity_out.append(fidelity)

    # Compute decay rate via linear regression
    decay_rate = 0.0
    if len(rounds_out) >= 2:
        x = np.array(rounds_out, dtype=float)
        y = np.array(fidelity_out, dtype=float)
        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        var_x = ((x - x_mean) ** 2).sum()
        if var_x > 0:
            decay_rate = float(((x - x_mean) * (y - y_mean)).sum() / var_x)

    # Compute half-life (first round where fidelity < 0.5)
    half_life = None
    for r, f in zip(rounds_out, fidelity_out):
        if f < 0.5:
            half_life = r
            break

    return {
        "rounds": rounds_out,
        "fidelity": fidelity_out,
        "decay_rate": decay_rate,
        "half_life": half_life,
    }


# ── Aggregate decay summary ─────────────────────────────────────────────


def compute_decay_summary(
    all_events: list[dict],
    agents: Iterable[Any],
    window: int = 5,
) -> dict[str, Any]:
    """Aggregate persona decay across all agents.

    Args:
        all_events: All events from the simulation (multi-agent).
        agents: Iterable of Agent objects (must have .profile).
        window: Sliding window size in rounds.

    Returns:
        {
            'per_agent': {agent_id: {rounds, fidelity, decay_rate, half_life}},
            'mean_fidelity_per_round': {round_id: float},
            'mean_decay_rate': float,
            'agents_drifted_pct': float,  # % with decay_rate < -0.01
        }
    """
    per_agent: dict[str, dict] = {}
    agents_list = list(agents)

    # Pre-filter events by agent
    events_by_agent: dict[str, list[dict]] = defaultdict(list)
    for e in all_events:
        aid = e.get("agent_id")
        if aid is not None:
            events_by_agent[aid].append(e)

    for agent in agents_list:
        aid = agent.profile.agent_id
        agent_events = events_by_agent.get(aid, [])
        per_agent[aid] = compute_per_round_persona_fidelity(
            agent_events, agent.profile, window=window
        )

    # Aggregate: mean fidelity per round
    round_fidelities: dict[int, list[float]] = defaultdict(list)
    for result in per_agent.values():
        for r, f in zip(result["rounds"], result["fidelity"]):
            round_fidelities[r].append(f)

    mean_fidelity_per_round = {
        r: float(np.mean(fs)) for r, fs in sorted(round_fidelities.items())
    }

    # Aggregate decay rate
    decay_rates = [r["decay_rate"] for r in per_agent.values() if r["rounds"]]
    mean_decay_rate = float(np.mean(decay_rates)) if decay_rates else 0.0

    # Drifted percentage: agents with decay_rate < -0.01
    n_agents = len(agents_list)
    n_drifted = sum(1 for r in decay_rates if r < -0.01)
    agents_drifted_pct = (n_drifted / n_agents * 100.0) if n_agents > 0 else 0.0

    return {
        "per_agent": per_agent,
        "mean_fidelity_per_round": mean_fidelity_per_round,
        "mean_decay_rate": mean_decay_rate,
        "agents_drifted_pct": agents_drifted_pct,
    }
