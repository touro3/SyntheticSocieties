"""Persona decay metrics — quantifying behavioral drift from initial persona.

Phase 24 — Limitations and failure mode analysis.

Measures how well an agent's behavior over time remains consistent with
the cooperation expectations set by its ESS-derived persona attributes.

Key concept:
  expected_cooperation_rate(profile) maps ESS-derived attributes to a
  baseline cooperation probability estimated from a logistic regression
  model fitted on real ESS Round 11 volunteering data (Austria, N=866).

  Persona fidelity at round r = 1 - |actual_coop_rate - expected_coop_rate|
  Decay rate = linear slope of fidelity over rounds (negative = drifting)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# ── Empirical cooperation rate model ─────────────────────────────────────

_MODEL_PATH = Path(__file__).parent.parent / "data" / "cooperation_model.json"

# Feature order must match scripts/fit_cooperation_model.py FEATURES list
_MODEL_FEATURES = [
    "trust_people",
    "trust_fairness",
    "trust_helpfulness",
    "risk_taking",
    "social_meeting_freq",
    "social_activity",
    "reduce_inequality",
]


@lru_cache(maxsize=1)
def _load_model() -> dict | None:
    """Load and cache the fitted cooperation model from JSON.

    Returns None if the model file is not found (falls back to heuristic).
    The lru_cache ensures the file is read only once per interpreter session.
    """
    if not _MODEL_PATH.exists():
        return None
    try:
        with _MODEL_PATH.open() as f:
            return json.load(f)
    except Exception:
        return None


def _logistic(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def expected_cooperation_rate(profile: Any) -> float:
    """Return the expected cooperation probability for a given agent profile.

    Uses a logistic regression model fitted on ESS Round 11 volunteering
    behavior (Austria, N=866, AUC=0.640, 10-fold CV).

    The model was fitted in ``scripts/fit_cooperation_model.py`` and stored
    in ``data/cooperation_model.json``. Volunteering is used as the
    cooperation proxy because it is the only directly observed altruistic
    behavior in the ESS (all other variables are attitudes or demographics).

    Feature → profile attribute mapping
    ------------------------------------
    trust_people       → profile.trust_people        (direct, or model mean)
    trust_fairness     → model training mean          (not in agent profile)
    trust_helpfulness  → model training mean          (not in agent profile)
    risk_taking        → profile.risk_tolerance       (same ESS variable)
    social_meeting_freq→ model training mean          (not in agent profile)
    social_activity    → profile.social_activity      (direct, or model mean)
    reduce_inequality  → model training mean          (not in agent profile)

    Attributes not in the profile are imputed with their training-data mean,
    which is conservative (predicts close to the population average for those
    dimensions). This imputation is documented here and in the model JSON.

    Known limitations
    -----------------
    - Austria-only data; cross-country generalization is untested
    - Volunteering ≠ in-game cooperation; effect sizes may differ
    - AUC=0.640 indicates moderate (not strong) predictive signal
    - ``trust_people`` is NOT a significant predictor in the fitted model
      (95% CI includes zero); ``risk_taking`` and social engagement drive
      the effect. This contradicts the prior heuristic formula's assumption.

    Falls back to heuristic formula if model file is missing.

    Args:
        profile: AgentProfile or any object with relevant attributes.

    Returns:
        Float in (0, 1): estimated P(cooperate | profile).
    """
    model = _load_model()
    if model is None:
        return _heuristic_cooperation_rate(profile)

    params = model["model_params"]
    coefs = params["coef_original"]  # coefficients on original scale
    intercept = params["intercept_original"]
    feat_means = params["feature_means"]  # training means for imputation

    # Build feature vector with imputation for missing profile attributes
    trust_people = getattr(profile, "trust_people", None)
    risk_tolerance = getattr(profile, "risk_tolerance", None)
    social_activity = getattr(profile, "social_activity", None)

    feat_vals = [
        trust_people if trust_people is not None else feat_means[0],  # trust_people
        feat_means[1],  # trust_fairness (imputed)
        feat_means[2],  # trust_helpfulness (imputed)
        risk_tolerance if risk_tolerance is not None else feat_means[3],  # risk_taking
        feat_means[4],  # social_meeting_freq (imputed)
        social_activity if social_activity is not None else feat_means[5],  # social_activity
        feat_means[6],  # reduce_inequality (imputed)
    ]

    log_odds = intercept + sum(c * x for c, x in zip(coefs, feat_vals))
    return _logistic(log_odds)


def _heuristic_cooperation_rate(profile: Any) -> float:
    """Fallback heuristic when ``data/cooperation_model.json`` is unavailable.

    This is a compact logistic regression that reproduces the *qualitative*
    findings of the Austrian ESS-11 fit (see ``data/cooperation_model.json``)
    without the dependency on that file. It is the path exercised in CI, where
    the fitted model JSON is gitignored.

    Empirical findings encoded here (all supported by bootstrap CIs that
    exclude zero in the full fit):
      * ``risk_tolerance``  — **positive** predictor (not negative as in the
        original toy formula ``0.2 + 0.6 * trust * (1 - risk)``)
      * ``social_activity`` — positive predictor; must actually appear in
        the equation so that high/low social agents produce different rates
      * ``trust_people``    — weak positive predictor (CI barely excludes 0)
      * Intercept tuned so the prediction surface stays inside
        ``[0.05, 0.45]`` across the full [0, 1]² (trust, risk) grid,
        centred near the 18 % Austrian volunteering base rate.
    """
    trust = getattr(profile, "trust_people", None)
    risk = getattr(profile, "risk_tolerance", None)
    social = getattr(profile, "social_activity", None)

    # Impute missing features with ESS-11 Austrian training-set means
    # (roughly: trust_people≈0.59, risk_taking≈0.30, social_activity≈0.30
    # after rescaling the 1-5 ESS sclact code into [0, 1]).
    if trust is None:
        trust = 0.5
    if risk is None:
        risk = 0.3
    if social is None:
        social = 0.3

    # Logistic on [0, 1] features. Coefficients chosen so that:
    #   min  at (t=0, r=0, s=0.3): sigmoid(-1.79) ≈ 0.143
    #   max  at (t=1, r=1, s=0.3): sigmoid(-0.49) ≈ 0.380
    # both safely inside the [0.05, 0.45] empirical-plausibility band.
    log_odds = -2.0 + 0.2 * trust + 1.1 * risk + 0.7 * social
    return _logistic(log_odds)


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
        per_agent[aid] = compute_per_round_persona_fidelity(agent_events, agent.profile, window=window)

    # Aggregate: mean fidelity per round
    round_fidelities: dict[int, list[float]] = defaultdict(list)
    for result in per_agent.values():
        for r, f in zip(result["rounds"], result["fidelity"]):
            round_fidelities[r].append(f)

    mean_fidelity_per_round = {r: float(np.mean(fs)) for r, fs in sorted(round_fidelities.items())}

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
