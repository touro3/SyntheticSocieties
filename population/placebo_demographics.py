"""Placebo persona generator — semantic-isolation control (Phase 1, audit response).

Research motivation
-------------------
The central BGF claim is that *empirical ESS grounding causally produces
behavioral realism*. A reviewer can object that any realism gain is a mere
artifact of **prompt heterogeneity / entropy**: grounded agents receive
varied, individuated personas, and that variety alone — not its sociological
*coherence* — could drive the behavioral spread we observe.

The placebo arm isolates the semantic component. It keeps the **demographic
skeleton** of real ESS respondents intact (age, gender, country, education,
income decile, social class) so prompts remain just as heterogeneous and
linguistically plausible as the grounded arm, but **independently permutes the
sociological trait vector across the population**. This preserves every
trait's *marginal* distribution (so the prompts look equally "ESS-like") while
destroying the *joint* correlation structure that grounding is supposed to
supply (e.g. the empirical coupling of low institutional trust with particular
political orientations). A persona may end up highly trustful yet violently
antisocial — structurally valid, sociologically nonsensical.

Contrast design (3 arms)
------------------------
    grounded      → ESS-coherent personas         (population.source: empirical)
    placebo       → scrambled-but-valid personas   (population.source: placebo)
    unconditioned → config-default synthetic       (population.source: synthetic)

If realism collapses from grounded → placebo, the gain is *semantic*. If it
survives, the gain was merely prompt entropy and the central claim is unsafe.

Implementation note
-------------------
We deliberately reuse :func:`population.persona_synthesizer.synthesize_ess_personas`
to draw a *coherent* cohort first, then scramble. This guarantees the placebo
marginals are byte-identical to the grounded marginals (the same sampled
values, only re-paired), which is exactly the property the isolation argument
needs. The permutation idiom mirrors ``population.generator._shuffle_traits``
(the existing "soul swap" Condition C) but is applied at the PersonaRecord
level so the rest of the pipeline is untouched.
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from agents.profile import _NORMALIZED_FIELDS
from population._helpers import clamp01 as _clamp01
from population.persona_synthesizer import PersonaRecord, synthesize_ess_personas

logger = logging.getLogger(__name__)

# Sociological / attitudinal traits whose joint structure encodes "grounding".
# Each is permuted *independently* (its own derived RNG stream) so that the
# population-level marginal of every field is preserved exactly while every
# pairwise correlation between them is destroyed.
#
# The demographic skeleton — agent_id, age, gender, country, education,
# education_level, income, income_decile, social_class, occupation, location,
# initial_wealth — is intentionally NOT in this list and stays row-aligned.
PLACEBO_SCRAMBLED_FIELDS: tuple[str, ...] = (
    "trust_people",
    "trust_institutions",
    "political_orientation",
    "life_satisfaction",
    "happiness",
    "immigration_attitude",
    "social_activity",
    "competitiveness",
    "religiosity",
)


def _clamp_normalized_fields(records: list[PersonaRecord]) -> None:
    """Clamp every [0,1]-normalized field in place.

    ``synthesize_ess_personas`` (unlike ``generate_empirical_population``) does
    not clamp, so raw ESS-derived values can fall outside [0,1] and would trip
    ``AgentProfile.__post_init__``. Clamping here makes the placebo arm
    pipeline-safe; it is applied *before* scrambling so the placebo marginals
    equal the clamped-grounded marginals exactly (the isolation property is
    stated against the clamped reference).
    """
    for rec in records:
        for field in _NORMALIZED_FIELDS:
            val = getattr(rec, field, None)
            if val is not None:
                setattr(rec, field, _clamp01(val))


def synthesize_placebo_personas(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    spec=None,
) -> list[PersonaRecord]:
    """Generate ``n`` structurally valid but sociologically scrambled personas.

    Args:
        df:   Cleaned ESS dataframe (e.g. ``data/ess_clean.parquet``).
        n:    Number of personas to synthesize.
        seed: Master seed. Each scrambled field gets its own derived seed so
              the permutations are mutually independent yet fully reproducible.
        spec: Optional ``SocietySpec`` forwarded to the underlying ESS
              synthesizer (currently unused by it; kept for signature parity).

    Returns:
        A list of :class:`PersonaRecord` with intact demographics and an
        independently shuffled sociological trait vector.

    The marginal distribution of every field in
    :data:`PLACEBO_SCRAMBLED_FIELDS` is identical to the grounded cohort drawn
    from the same ``df``/``seed``; only the joint structure is broken.
    """
    base_records = synthesize_ess_personas(df, spec, n, seed=seed)
    _clamp_normalized_fields(base_records)

    if len(base_records) < 2:
        logger.warning(
            "Placebo scramble skipped: need >=2 personas to permute (got %d). Returning the coherent cohort unchanged.",
            len(base_records),
        )
        return base_records

    master = random.Random(seed)
    field_seeds = {f: master.randint(0, 2**31 - 1) for f in PLACEBO_SCRAMBLED_FIELDS}

    for field in PLACEBO_SCRAMBLED_FIELDS:
        values = [getattr(rec, field) for rec in base_records]
        # Independent permutation per field → marginal preserved, joints broken.
        random.Random(field_seeds[field]).shuffle(values)
        for rec, val in zip(base_records, values):
            setattr(rec, field, val)

    logger.info(
        "Placebo cohort built: n=%d, scrambled %d sociological fields "
        "(marginals preserved, joint structure destroyed). seed=%d",
        len(base_records),
        len(PLACEBO_SCRAMBLED_FIELDS),
        seed,
    )
    return base_records
