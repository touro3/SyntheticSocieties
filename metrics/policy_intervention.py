"""Policy Intervention Analysis — Phase 28.8.

Models trust-building policy interventions applied mid-simulation.
A "trust boost" of intensity δ raises every agent's effective trust_people
by δ starting at the intervention round, shifting cooperation rates upward.

This converts BGF from a "research tool" into a "policy analysis platform":
researchers can identify the optimal intervention intensity, timing, and
expected return in terms of cooperation, wealth, and inequality outcomes.

Key measures per (intensity, seed):
    cooperation_rate_pre  : mean cooperation rate rounds [0, intervention_round)
    cooperation_rate_post : mean cooperation rate rounds [intervention_round, n_rounds)
    delta_cooperation     : post − pre
    wealth_mean_final     : mean wealth at final round
    gini_final            : Gini coefficient at final round

Usage
-----
    from metrics.policy_intervention import run_intervention_sweep
    results = run_intervention_sweep()
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class InterventionResult:
    """Outcome of one (intensity, seed) intervention experiment."""

    intensity: float           # Trust boost δ — e.g. 0.0, 0.05, 0.10, 0.20
    intervention_round: int    # Round at which the boost is applied
    n_agents: int
    n_rounds: int
    seed: int

    cooperation_rate_pre: float    # Mean cooperation rate before intervention
    cooperation_rate_post: float   # Mean cooperation rate after intervention
    delta_cooperation: float       # post − pre
    wealth_mean_final: float       # Mean wealth at last round
    gini_final: float              # Gini coefficient at last round

    per_round_cooperation: list[float] = field(default_factory=list)
    per_round_wealth_mean: list[float] = field(default_factory=list)


@dataclass
class InterventionSummary:
    """Aggregated results across seeds for one intensity level."""

    intensity: float
    intensity_pct: str          # e.g. "10%"
    coop_pre_mean: float
    coop_post_mean: float
    delta_coop_mean: float
    delta_coop_std: float
    gini_mean: float
    wealth_mean_final: float
    per_round_cooperation: list[float]   # from first seed, for plotting


# ── Internal helpers ───────────────────────────────────────────────────────


def _hash_uniform(agent_id: str, round_id: int) -> float:
    """Deterministic pseudo-random float in [0, 1) — reproducible across runs."""
    key = f"{agent_id}:{round_id}".encode()
    digest = hashlib.sha256(key).digest()
    uint32 = struct.unpack(">I", digest[:4])[0]
    return uint32 / 4_294_967_296.0


def _gini(values: list[float]) -> float:
    """Gini coefficient using the canonical sorted-array formula."""
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    cumsum = sum(arr)
    if cumsum == 0.0:
        return 0.0
    numerator = sum((i + 1) * v for i, v in enumerate(arr))
    return (2.0 * numerator) / (n * cumsum) - (n + 1.0) / n


# ── Core simulation ────────────────────────────────────────────────────────


def run_single_intervention(
    intensity: float,
    intervention_round: int = 15,
    n_agents: int = 200,
    n_rounds: int = 30,
    seed: int = 42,
) -> InterventionResult:
    """Run one intervention experiment.

    Agents follow a rule-based ESS-grounded cooperation probability
    (mirrors RuleBasedESSPolicy). At ``intervention_round``, effective
    trust_people is boosted by ``intensity`` (clamped to [0, 1]).

    Args:
        intensity: Trust-boost δ — e.g. 0.05 = +5 percentage points.
        intervention_round: Round at which policy takes effect (0-indexed).
        n_agents: Population size.
        n_rounds: Total simulation length.
        seed: RNG seed for reproducibility.

    Returns:
        InterventionResult with pre/post cooperation rates, wealth, Gini.
    """
    rng = np.random.default_rng(seed)

    # Sample heterogeneous agent profiles
    trust_base = rng.beta(2, 2, size=n_agents)      # ESS-realistic: centered ~0.5
    risk = rng.beta(2, 3, size=n_agents)             # Slightly risk-averse
    social = rng.beta(2, 2, size=n_agents)           # Social activity
    wealth = np.full(n_agents, 100.0, dtype=float)   # Starting wealth
    agent_ids = [f"agent_{i:04d}" for i in range(n_agents)]

    per_round_coop: list[float] = []
    per_round_wealth: list[float] = []

    for r in range(n_rounds):
        # Apply trust boost at intervention round
        effective_trust = np.clip(
            trust_base + (intensity if r >= intervention_round else 0.0),
            0.0, 1.0,
        )

        # Cooperation probability — mirrors RuleBasedESSPolicy formula
        coop_prob = np.clip(
            0.2 + 0.5 * effective_trust * (1.0 - risk) + 0.15 * social,
            0.05, 0.90,
        )

        cooperated = np.array([
            1 if _hash_uniform(agent_ids[i], r) < coop_prob[i] else 0
            for i in range(n_agents)
        ])

        per_round_coop.append(float(cooperated.mean()))

        # Economy: cooperation yields net surplus (+7 vs +10 for work);
        # the deficit reflects the real cost of cooperation (-3 to self,
        # +12/cooperator pair net). Here we simplify to per-agent net.
        for i in range(n_agents):
            wealth[i] += 7.0 if cooperated[i] else 10.0

        per_round_wealth.append(float(wealth.mean()))

    pre_rounds = per_round_coop[:intervention_round] if intervention_round > 0 else []
    post_rounds = per_round_coop[intervention_round:]

    coop_pre = float(np.mean(pre_rounds)) if pre_rounds else float(per_round_coop[0])
    coop_post = float(np.mean(post_rounds)) if post_rounds else float(per_round_coop[-1])

    return InterventionResult(
        intensity=intensity,
        intervention_round=intervention_round,
        n_agents=n_agents,
        n_rounds=n_rounds,
        seed=seed,
        cooperation_rate_pre=coop_pre,
        cooperation_rate_post=coop_post,
        delta_cooperation=coop_post - coop_pre,
        wealth_mean_final=float(wealth.mean()),
        gini_final=_gini(wealth.tolist()),
        per_round_cooperation=per_round_coop,
        per_round_wealth_mean=per_round_wealth,
    )


def run_intervention_sweep(
    intensities: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    intervention_round: int = 15,
    n_agents: int = 200,
    n_rounds: int = 30,
    seeds: tuple[int, ...] = (42, 123, 7),
) -> list[InterventionResult]:
    """Run full sweep across intensities × seeds.

    Args:
        intensities: Trust-boost values to test.
        intervention_round: Round at which policy is applied.
        n_agents: Population size per run.
        n_rounds: Total rounds per run.
        seeds: RNG seeds — one run per seed per intensity.

    Returns:
        Flat list of InterventionResult, length = len(intensities) × len(seeds).
    """
    results: list[InterventionResult] = []
    for intensity in intensities:
        for seed in seeds:
            results.append(
                run_single_intervention(
                    intensity=intensity,
                    intervention_round=intervention_round,
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    seed=seed,
                )
            )
    return results


def aggregate_by_intensity(
    results: list[InterventionResult],
) -> list[InterventionSummary]:
    """Aggregate InterventionResults across seeds, grouped by intensity.

    Args:
        results: Raw results from run_intervention_sweep().

    Returns:
        List of InterventionSummary, one per intensity level, sorted ascending.
    """
    from collections import defaultdict

    by_intensity: dict[float, list[InterventionResult]] = defaultdict(list)
    for r in results:
        by_intensity[r.intensity].append(r)

    summaries: list[InterventionSummary] = []
    for intensity in sorted(by_intensity):
        group = by_intensity[intensity]
        delta_coops = [r.delta_cooperation for r in group]
        summaries.append(
            InterventionSummary(
                intensity=intensity,
                intensity_pct=f"{intensity:.0%}",
                coop_pre_mean=float(np.mean([r.cooperation_rate_pre for r in group])),
                coop_post_mean=float(np.mean([r.cooperation_rate_post for r in group])),
                delta_coop_mean=float(np.mean(delta_coops)),
                delta_coop_std=float(np.std(delta_coops)),
                gini_mean=float(np.mean([r.gini_final for r in group])),
                wealth_mean_final=float(np.mean([r.wealth_mean_final for r in group])),
                per_round_cooperation=group[0].per_round_cooperation,
            )
        )
    return summaries
