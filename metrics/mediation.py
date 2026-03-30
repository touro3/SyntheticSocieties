"""Mediation analysis — decomposing grounding effects into components.

Phase 19 — Causal inference and ablation formalization.

Decomposes the total effect of BGF grounding (Condition B vs baseline)
into:
  - Persona effect:     persona conditioning alone
  - RAG effect:         retrieval context alone
  - Interaction effect: synergy (or interference) between persona + RAG

This follows a standard 2x2 factorial decomposition:

    Total effect       = full_grounded - baseline
    Persona effect     = persona_only - baseline
    RAG effect         = rag_only - baseline
    Interaction effect = total - persona - rag
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mediation_decomposition(
    full_grounded_coop: float,
    persona_only_coop: float,
    rag_only_coop: float,
    baseline_coop: float,
) -> dict[str, float]:
    """Decompose the total grounding effect into persona, RAG, and interaction.

    Args:
        full_grounded_coop: Cooperation rate with persona + RAG (Condition B).
        persona_only_coop: Cooperation rate with persona only (no RAG).
        rag_only_coop: Cooperation rate with RAG only (no persona).
        baseline_coop: Cooperation rate with neither (Condition A baseline).

    Returns:
        Dict with total, persona, RAG, and interaction effects, plus
        persona and RAG shares of the total effect.
    """
    total = full_grounded_coop - baseline_coop
    persona = persona_only_coop - baseline_coop
    rag = rag_only_coop - baseline_coop
    interaction = total - persona - rag

    if abs(total) > 1e-10:
        persona_share = persona / total
        rag_share = rag / total
    else:
        persona_share = 0.0
        rag_share = 0.0

    return {
        "total_effect": total,
        "persona_effect": persona,
        "rag_effect": rag,
        "interaction_effect": interaction,
        "persona_share": persona_share,
        "rag_share": rag_share,
    }


def mediation_table(
    conditions: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build a mediation analysis table from experimental conditions.

    Args:
        conditions: Dict mapping condition name to metric dict.
            Required keys: 'baseline', 'persona_only', 'rag_only',
            'full_grounded'. Each value is a dict of {metric_name: value}.

    Returns:
        DataFrame with rows = metrics, columns = effects.
    """
    baseline = conditions["baseline"]
    persona_only = conditions["persona_only"]
    rag_only = conditions["rag_only"]
    full_grounded = conditions["full_grounded"]

    # Get all metric names from the baseline
    metric_names = sorted(baseline.keys())

    rows = []
    for metric in metric_names:
        decomp = compute_mediation_decomposition(
            full_grounded_coop=full_grounded[metric],
            persona_only_coop=persona_only[metric],
            rag_only_coop=rag_only[metric],
            baseline_coop=baseline[metric],
        )
        decomp["metric"] = metric
        rows.append(decomp)

    df = pd.DataFrame(rows)
    df = df.set_index("metric")
    return df


def bootstrap_mediation_decomposition(
    full_grounded_samples: list[float],
    persona_only_samples: list[float],
    rag_only_samples: list[float],
    baseline_samples: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence intervals for mediation decomposition effects.

    Resamples each condition independently, computes the decomposition for
    each bootstrap replicate, and returns point estimates with CIs for
    total, persona, RAG, and interaction effects.

    Args:
        full_grounded_samples: Per-seed cooperation rates for full grounded.
        persona_only_samples: Per-seed cooperation rates for persona-only.
        rag_only_samples: Per-seed cooperation rates for RAG-only.
        baseline_samples: Per-seed cooperation rates for baseline.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (0-1).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with point estimates and CI bounds for each effect.
    """
    rng = np.random.RandomState(seed)

    full = np.asarray(full_grounded_samples, dtype=float)
    persona = np.asarray(persona_only_samples, dtype=float)
    rag = np.asarray(rag_only_samples, dtype=float)
    base = np.asarray(baseline_samples, dtype=float)

    # Point estimates
    point = compute_mediation_decomposition(
        full_grounded_coop=float(np.mean(full)),
        persona_only_coop=float(np.mean(persona)),
        rag_only_coop=float(np.mean(rag)),
        baseline_coop=float(np.mean(base)),
    )

    # Bootstrap resampling
    effects = {k: [] for k in ["total_effect", "persona_effect", "rag_effect", "interaction_effect"]}
    for _ in range(n_bootstrap):
        b_full = rng.choice(full, size=len(full), replace=True)
        b_persona = rng.choice(persona, size=len(persona), replace=True)
        b_rag = rng.choice(rag, size=len(rag), replace=True)
        b_base = rng.choice(base, size=len(base), replace=True)

        decomp = compute_mediation_decomposition(
            full_grounded_coop=float(np.mean(b_full)),
            persona_only_coop=float(np.mean(b_persona)),
            rag_only_coop=float(np.mean(b_rag)),
            baseline_coop=float(np.mean(b_base)),
        )
        for k in effects:
            effects[k].append(decomp[k])

    alpha = 1.0 - confidence
    result = {}
    for k in effects:
        arr = np.array(effects[k])
        result[k] = point[k]
        result[f"{k}_ci_lower"] = float(np.percentile(arr, 100 * alpha / 2))
        result[f"{k}_ci_upper"] = float(np.percentile(arr, 100 * (1 - alpha / 2)))

    # Include shares from point estimate
    result["persona_share"] = point["persona_share"]
    result["rag_share"] = point["rag_share"]

    return result
