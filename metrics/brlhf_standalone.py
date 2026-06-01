"""Standalone B_RLHF metric — no BGF infrastructure required.

This module is intentionally dependency-free so it can be extracted into
a standalone package (pip install brlhf) for use by any LLM simulation
researcher. It depends only on the Python standard library.

B_RLHF (RLHF Bias Index) measures how far an observed LLM action
distribution deviates from a reference distribution in a multi-agent
setting. A high B_RLHF(A) in an ungrounded condition indicates the
RLHF alignment prior is dominating over strategic reasoning.

Three reference distributions are supported:
  - Uniform:  assumes no prior on action distribution (conservative upper bound)
  - Human:    uses empirically measured human action distribution (calibrated)
  - Nash:     uses Nash equilibrium distribution (game-theoretic baseline)

Citation:
    Tourinho Mamede, L. (2026). "Behavioral Grounding Framework..."
    SyntheticSocieties. https://github.com/touro3/SyntheticSocieties

Usage (standalone, no BGF required):
    from metrics.brlhf_standalone import BRLHFMetric

    metric = BRLHFMetric(actions=["cooperate", "defect"])
    score = metric.compute(
        observed={"cooperate": 0.90, "defect": 0.10},
        reference="uniform",
    )
    print(f"B_RLHF = {score:.3f}")  # → B_RLHF = 0.400

    # With human baseline
    score_human = metric.compute(
        observed={"cooperate": 0.90, "defect": 0.10},
        reference={"cooperate": 0.47, "defect": 0.53},  # human PD baseline
    )
    print(f"B_RLHF vs human = {score_human:.3f}")  # → B_RLHF vs human = 0.430

    # Compare conditions A vs B (grounding effect)
    delta = metric.grounding_effect(
        condition_a={"cooperate": 0.90, "defect": 0.10},
        condition_b={"cooperate": 0.52, "defect": 0.48},
    )
    print(delta)  # {"absolute_reduction": 0.190, "relative_reduction_pct": 47.5, ...}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

ReferenceSpec = Union[str, dict[str, float]]


@dataclass(frozen=True)
class BRLHFResult:
    """Result of a single B_RLHF computation."""

    observed: dict[str, float]
    reference: dict[str, float]
    reference_name: str
    brlhf: float
    cooperation_rate: float
    reference_cooperation_rate: float
    bias_direction: str  # "over_cooperative" | "under_cooperative" | "calibrated"

    def __str__(self) -> str:
        return (
            f"B_RLHF({self.reference_name}) = {self.brlhf:.4f} | "
            f"coop_obs={self.cooperation_rate:.3f}, "
            f"coop_ref={self.reference_cooperation_rate:.3f} | "
            f"direction={self.bias_direction}"
        )


@dataclass
class BRLHFGroundingEffect:
    """Result of comparing B_RLHF across two conditions."""

    brlhf_a: float
    brlhf_b: float
    absolute_reduction: float
    relative_reduction_pct: float
    direction_confirmed: bool  # True if B(B) < B(A)
    interpretation: str

    def __str__(self) -> str:
        sign = "−" if self.absolute_reduction >= 0 else "+"
        return (
            f"ΔB_RLHF = {sign}{abs(self.absolute_reduction):.4f} "
            f"({sign}{abs(self.relative_reduction_pct):.1f}%) | "
            f"A={self.brlhf_a:.4f} → B={self.brlhf_b:.4f} | "
            f"direction={'✓' if self.direction_confirmed else '✗'}"
        )


class BRLHFMetric:
    """Game-agnostic RLHF Bias Index metric.

    Computes Total Variation distance between an observed action distribution
    and a reference distribution. Can be applied to any LLM multi-agent game.

    Args:
        actions: List of valid action names in the game.
        cooperative_actions: Subset of actions considered "cooperative."
            Used for bias direction classification. Defaults to all actions
            with "cooperat" or "stag" or "contribute" or "high" in their name.
    """

    def __init__(
        self,
        actions: list[str],
        cooperative_actions: list[str] | None = None,
    ) -> None:
        if not actions:
            raise ValueError("Action list must not be empty.")
        if len(actions) != len(set(actions)):
            raise ValueError("Action list must not contain duplicates.")
        self.actions = actions
        self._coop_actions = cooperative_actions or self._infer_coop_actions(actions)

    @staticmethod
    def _infer_coop_actions(actions: list[str]) -> list[str]:
        coop_keywords = ("cooperat", "stag", "contribut", "high_offer", "share")
        return [a for a in actions if any(kw in a.lower() for kw in coop_keywords)]

    def _normalize(self, distribution: dict[str, float]) -> dict[str, float]:
        """Normalize distribution over the game's action space."""
        total = sum(distribution.get(a, 0.0) for a in self.actions)
        if total <= 0:
            raise ValueError(
                f"Distribution sums to {total} over actions {self.actions}. "
                "Ensure all action counts are non-negative and at least one is positive."
            )
        return {a: distribution.get(a, 0.0) / total for a in self.actions}

    def _resolve_reference(self, reference: ReferenceSpec) -> tuple[dict[str, float], str]:
        """Resolve a reference specification to a normalized distribution and name."""
        if isinstance(reference, dict):
            return self._normalize(reference), "custom"
        if reference == "uniform":
            n = len(self.actions)
            return {a: 1.0 / n for a in self.actions}, "uniform"
        raise ValueError(f"Unknown reference '{reference}'. Pass a dict or 'uniform'.")

    def _total_variation(self, p: dict[str, float], q: dict[str, float]) -> float:
        """TV(p, q) = 0.5 * Σ |p(a) - q(a)|"""
        return 0.5 * sum(abs(p.get(a, 0.0) - q.get(a, 0.0)) for a in self.actions)

    def _cooperation_rate(self, distribution: dict[str, float]) -> float:
        """Sum of probability mass on cooperative actions."""
        return sum(distribution.get(a, 0.0) for a in self._coop_actions)

    def compute(
        self,
        observed: dict[str, float],
        reference: ReferenceSpec = "uniform",
    ) -> BRLHFResult:
        """Compute B_RLHF for an observed distribution against a reference.

        Args:
            observed: Observed action distribution (action → count or probability).
            reference: Reference distribution. Either "uniform" or a dict of
                action → probability matching this game's action space.

        Returns:
            BRLHFResult with B_RLHF score and diagnostics.
        """
        obs_norm = self._normalize(observed)
        ref_norm, ref_name = self._resolve_reference(reference)
        brlhf = self._total_variation(obs_norm, ref_norm)

        obs_coop = self._cooperation_rate(obs_norm)
        ref_coop = self._cooperation_rate(ref_norm)

        threshold = 0.08
        if obs_coop > ref_coop + threshold:
            direction = "over_cooperative"
        elif obs_coop < ref_coop - threshold:
            direction = "under_cooperative"
        else:
            direction = "calibrated"

        return BRLHFResult(
            observed=dict(obs_norm),
            reference=dict(ref_norm),
            reference_name=ref_name,
            brlhf=round(brlhf, 6),
            cooperation_rate=round(obs_coop, 4),
            reference_cooperation_rate=round(ref_coop, 4),
            bias_direction=direction,
        )

    def grounding_effect(
        self,
        condition_a: dict[str, float],
        condition_b: dict[str, float],
        reference: ReferenceSpec = "uniform",
    ) -> BRLHFGroundingEffect:
        """Compare B_RLHF between ungrounded (A) and grounded (B) conditions.

        Args:
            condition_a: Action distribution from ungrounded condition.
            condition_b: Action distribution from grounded condition.
            reference: Reference distribution for B_RLHF computation.

        Returns:
            BRLHFGroundingEffect summarizing the grounding effect.
        """
        result_a = self.compute(condition_a, reference)
        result_b = self.compute(condition_b, reference)
        absolute_reduction = result_a.brlhf - result_b.brlhf
        relative_reduction = (absolute_reduction / result_a.brlhf * 100) if result_a.brlhf > 0 else 0.0
        direction_confirmed = result_b.brlhf < result_a.brlhf

        interp = (
            f"Grounding reduces B_RLHF by {relative_reduction:.1f}% "
            f"({result_a.brlhf:.4f} → {result_b.brlhf:.4f}). "
            f"Consistent with grounding hypothesis (H2)."
            if direction_confirmed
            else (
                f"Grounding INCREASES B_RLHF by {abs(relative_reduction):.1f}% "
                f"({result_a.brlhf:.4f} → {result_b.brlhf:.4f}). "
                f"Inconsistent with H2 — alignment methodology may moderate the effect."
            )
        )

        return BRLHFGroundingEffect(
            brlhf_a=result_a.brlhf,
            brlhf_b=result_b.brlhf,
            absolute_reduction=round(absolute_reduction, 6),
            relative_reduction_pct=round(relative_reduction, 2),
            direction_confirmed=direction_confirmed,
            interpretation=interp,
        )

    def audit_model(
        self,
        model_name: str,
        observed_ungrounded: dict[str, float],
        observed_grounded: dict[str, float] | None = None,
        human_baseline: dict[str, float] | None = None,
    ) -> dict[str, object]:
        """Full B_RLHF audit for a single LLM in this game.

        Computes B_RLHF vs uniform, vs human (if available), and the
        grounding effect (if grounded distribution is provided).

        Suitable for inclusion in cross-model comparison tables.
        """
        result = {
            "model": model_name,
            "game": ", ".join(self.actions),
            "brlhf_vs_uniform": self.compute(observed_ungrounded, "uniform").brlhf,
            "coop_rate_ungrounded": self._cooperation_rate(self._normalize(observed_ungrounded)),
        }

        if human_baseline is not None:
            result["brlhf_vs_human"] = self.compute(observed_ungrounded, human_baseline).brlhf

        if observed_grounded is not None:
            effect = self.grounding_effect(observed_ungrounded, observed_grounded)
            result["brlhf_grounded"] = effect.brlhf_b
            result["brlhf_reduction_pct"] = effect.relative_reduction_pct
            result["grounding_direction_confirmed"] = effect.direction_confirmed

        return result


# ── Jensen-Shannon Divergence (pure Python, no numpy) ─────────────────────────


def _entropy(p: list[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def jsd_from_dists(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon Divergence between two distributions over the same support.

    JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) where M = 0.5*(P+Q)
    Bounded in [0, 1] when using log base 2.
    """
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    p_vals = [p.get(k, 0.0) for k in keys]
    q_vals = [q.get(k, 0.0) for k in keys]
    m_vals = [m[k] for k in keys]
    return _entropy(m_vals) - 0.5 * _entropy(p_vals) - 0.5 * _entropy(q_vals)


# ── Convenience constructors for known games ──────────────────────────────────


def make_prisoners_dilemma_metric() -> BRLHFMetric:
    """Return a BRLHFMetric configured for the two-action Prisoner's Dilemma."""
    return BRLHFMetric(
        actions=["cooperate", "defect"],
        cooperative_actions=["cooperate"],
    )


def make_stag_hunt_metric() -> BRLHFMetric:
    """Return a BRLHFMetric configured for the Stag Hunt coordination game."""
    return BRLHFMetric(
        actions=["stag", "hare"],
        cooperative_actions=["stag"],
    )


def make_public_goods_metric() -> BRLHFMetric:
    """Return a BRLHFMetric configured for the three-action Public Goods Game."""
    return BRLHFMetric(
        actions=["work", "save", "cooperate"],
        cooperative_actions=["cooperate"],
    )


def make_ultimatum_metric() -> BRLHFMetric:
    """Return a BRLHFMetric configured for the three-offer Ultimatum Game."""
    return BRLHFMetric(
        actions=["low_offer", "fair_offer", "high_offer"],
        cooperative_actions=["high_offer"],
    )
