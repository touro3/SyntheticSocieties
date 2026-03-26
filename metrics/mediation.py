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
