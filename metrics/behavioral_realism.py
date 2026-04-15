"""Behavioral Realism Metric (BRM) and Action Concentration Index.

Phase 23 — Formal framework formalization.

Provides three core metrics for evaluating how well a simulation
matches empirical data:

  BRM_JSD:  1 - JSD(D_sim || D_ESS)                   ∈ [0, 1]
  B_RLHF:   TV(π_observed, π_uniform)                  ∈ [0, 1]
  Composite BRM:  weighted aggregate of sub-dimensions  ∈ [0, 1]

All metrics are oriented so that 1 = better / more realistic.

Interpretation note — B_RLHF:
  B_RLHF measures total variation distance from a *uniform* action
  distribution. In this project it is used as a diagnostic for whether
  RLHF fine-tuning has steered the base model toward a particular action
  (typically cooperation/helpfulness), producing a skewed distribution.
  The hypothesis is that an ungrounded RLHF-tuned LLM will over-cooperate
  (high B_RLHF), while ESS grounding pushes action distributions toward
  empirically realistic heterogeneity (lower B_RLHF).

  Limitation: B_RLHF cannot distinguish RLHF-induced concentration from
  legitimate persona-driven concentration (e.g. a cohort of high-trust
  agents will naturally cooperate more). Interpret B_RLHF differences
  between Condition A and B as evidence of grounding effect, not as a
  standalone measure of RLHF bias.
"""

from __future__ import annotations

from collections.abc import Iterable

from metrics.distribution import jensen_shannon_divergence

# ── Actions recognized by the BGF action space ───────────────────────────

_ACTIONS = ("work", "save", "cooperate")
_UNIFORM = 1.0 / len(_ACTIONS)


# ── BRM-JSD ──────────────────────────────────────────────────────────────


def compute_brm_jsd(
    sim_values: Iterable[float],
    emp_values: Iterable[float],
    bins: int = 30,
) -> float:
    """Behavioral Realism Metric based on Jensen-Shannon Divergence.

    BRM_JSD = 1 - JSD(D_sim || D_ESS)

    Args:
        sim_values: Simulated distribution (e.g. wealth array).
        emp_values: Empirical target distribution (e.g. ESS wealth).
        bins: Number of histogram bins for JSD computation.

    Returns:
        Float in [0, 1]. 1 = identical distributions (perfect realism).

    Raises:
        ValueError: If either distribution is empty.
    """
    jsd = jensen_shannon_divergence(sim_values, emp_values, bins=bins)
    return max(0.0, min(1.0, 1.0 - jsd))


# ── RLHF Bias Index ─────────────────────────────────────────────────────


def compute_rlhf_bias_index(action_distribution: dict[str, float]) -> float:
    """Action Concentration Index via Total Variation distance from uniform.

    Also referred to as B_RLHF in the paper (see module docstring for
    interpretation and limitations before using this for causal claims).

    B_RLHF(π) = 0.5 * Σ |π(a) - 1/|A||  for a ∈ {work, save, cooperate}

    This equals the total variation distance between the observed action
    distribution and a uniform distribution over the action space. A value
    of 0 means actions are equally distributed; 2/3 means all agents take
    the same action (maximum concentration).

    Args:
        action_distribution: Mapping of action name → probability.
            Missing actions are treated as 0.0.

    Returns:
        Float in [0, 2/3]. 0 = uniform (no concentration), 2/3 = fully concentrated.

    Raises:
        ValueError: If distribution is empty.
    """
    if not action_distribution:
        raise ValueError("Action distribution must not be empty.")

    tv = 0.0
    for action in _ACTIONS:
        p = action_distribution.get(action, 0.0)
        tv += abs(p - _UNIFORM)

    return 0.5 * tv


def rlhf_bias_index_from_counts(action_counts: dict[str, int]) -> float:
    """Compute RLHF Bias Index from raw action counts.

    Normalizes counts to a probability distribution, then delegates
    to :func:`compute_rlhf_bias_index`.

    Raises:
        ValueError: If total count is zero or dict is empty.
    """
    if not action_counts:
        raise ValueError("Action counts must not be empty.")

    total = sum(action_counts.values())
    if total == 0:
        raise ValueError("Total action count is zero — cannot normalize.")

    dist = {k: v / total for k, v in action_counts.items()}
    return compute_rlhf_bias_index(dist)


# ── Composite BRM ────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS = {
    "jsd": 0.30,
    "gini_gap": 0.25,
    "coop_gap": 0.25,
    "stability": 0.20,
}


def compute_composite_brm(
    sim_wealth: Iterable[float],
    emp_wealth: Iterable[float],
    sim_gini: float,
    emp_gini: float,
    sim_coop_rate: float,
    emp_coop_rate: float,
    temporal_stability_jsd: float,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Composite Behavioral Realism Metric aggregating sub-dimensions.

    Each component is normalized to [0, 1] where 1 = better, then
    combined via a weighted average.

    Components:
        jsd_component:       1 - JSD(wealth_sim, wealth_ESS)
        gini_component:      1 - |sim_gini - emp_gini|
        coop_component:      1 - |sim_coop - emp_coop|
        stability_component: 1 - temporal_stability_jsd

    Args:
        sim_wealth: Simulated wealth distribution.
        emp_wealth: Empirical wealth distribution.
        sim_gini: Simulated Gini coefficient.
        emp_gini: Empirical Gini coefficient.
        sim_coop_rate: Simulated cooperation rate.
        emp_coop_rate: Empirical cooperation rate.
        temporal_stability_jsd: Mean round-to-round JSD (lower = more stable).
        weights: Optional weight dict. Keys must match _DEFAULT_WEIGHTS.
            Must sum to 1.0 (±0.01 tolerance).

    Returns:
        Dict with 'composite' and per-component scores, all in [0, 1].

    Raises:
        ValueError: If weights do not sum to 1.
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS

    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0 (got {weight_sum:.4f}).")

    jsd_comp = compute_brm_jsd(sim_wealth, emp_wealth)
    gini_comp = max(0.0, min(1.0, 1.0 - abs(sim_gini - emp_gini)))
    coop_comp = max(0.0, min(1.0, 1.0 - abs(sim_coop_rate - emp_coop_rate)))
    stab_comp = max(0.0, min(1.0, 1.0 - temporal_stability_jsd))

    composite = (
        w.get("jsd", 0.0) * jsd_comp
        + w.get("gini_gap", 0.0) * gini_comp
        + w.get("coop_gap", 0.0) * coop_comp
        + w.get("stability", 0.0) * stab_comp
    )

    return {
        "composite": max(0.0, min(1.0, composite)),
        "jsd_component": jsd_comp,
        "gini_component": gini_comp,
        "coop_component": coop_comp,
        "stability_component": stab_comp,
    }
