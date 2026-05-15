"""Tests for the placebo / semantic-isolation persona generator (Phase 1).

These guarantee the placebo arm has exactly the properties the audit-response
argument requires:

  (a) demographic skeleton preserved row-for-row vs the grounded cohort,
  (b) every scrambled trait's MARGINAL distribution is identical to grounded,
  (c) the JOINT correlation structure is destroyed (the isolation property),
  (d) fully deterministic under a fixed seed,
  (e) the records still build valid AgentProfile objects (pipeline-safe).

All tests run on the real data/ess_clean.parquet — no LLM, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decision.mock_policy import MockPolicy
from population._helpers import clamp01
from population.persona_synthesizer import persona_records_to_agents, synthesize_ess_personas
from population.placebo_demographics import PLACEBO_SCRAMBLED_FIELDS, synthesize_placebo_personas

ESS_PARQUET = Path(__file__).resolve().parents[1] / "data" / "ess_clean.parquet"

DEMOGRAPHIC_FIELDS = (
    "agent_id",
    "age",
    "gender",
    "country",
    "education",
    "education_level",
    "income",
    "income_decile",
    "social_class",
    "occupation",
    "location",
    "initial_wealth",
)

N = 120
SEED = 42


@pytest.fixture(scope="module")
def ess_df() -> pd.DataFrame:
    if not ESS_PARQUET.exists():
        pytest.skip(f"ESS parquet not available at {ESS_PARQUET}")
    return pd.read_parquet(ESS_PARQUET)


@pytest.fixture(scope="module")
def grounded(ess_df):
    return synthesize_ess_personas(ess_df, None, N, seed=SEED)


@pytest.fixture(scope="module")
def placebo(ess_df):
    return synthesize_placebo_personas(ess_df, n=N, seed=SEED)


def test_demographics_preserved_rowwise(grounded, placebo):
    """(a) Same seed ⇒ identical demographic skeleton per agent."""
    assert len(placebo) == len(grounded) == N
    for g, p in zip(grounded, placebo):
        for field in DEMOGRAPHIC_FIELDS:
            assert getattr(g, field) == getattr(p, field), (
                f"demographic field {field!r} changed: {getattr(g, field)} -> {getattr(p, field)}"
            )


def test_scrambled_trait_marginals_preserved(grounded, placebo):
    """(b) Each scrambled field is a permutation of the clamped-grounded values.

    The isolation property is stated against the *clamped* grounded cohort,
    since the placebo arm clamps to [0,1] for pipeline safety before scrambling.
    """
    for field in PLACEBO_SCRAMBLED_FIELDS:
        g_vals = sorted(clamp01(getattr(r, field)) for r in grounded if getattr(r, field) is not None)
        p_vals = sorted(getattr(r, field) for r in placebo if getattr(r, field) is not None)
        assert g_vals == p_vals, f"marginal of {field!r} not preserved under scramble"


def test_joint_structure_destroyed(grounded, placebo):
    """(c) An empirically coupled trait pair decorrelates under scrambling."""

    def corr(records, a, b):
        pairs = [
            (getattr(r, a), getattr(r, b)) for r in records if getattr(r, a) is not None and getattr(r, b) is not None
        ]
        s = pd.DataFrame(pairs, columns=["a", "b"])
        return float(s["a"].corr(s["b"]))

    g_corr = abs(corr(grounded, "trust_people", "trust_institutions"))
    p_corr = abs(corr(placebo, "trust_people", "trust_institutions"))

    # Grounded carries real ESS coupling; placebo should be markedly weaker.
    assert g_corr > 0.1, f"sanity: expected real ESS coupling, got {g_corr:.3f}"
    assert p_corr < g_corr, f"scramble failed to break joint structure ({p_corr:.3f} !< {g_corr:.3f})"


def test_deterministic_under_fixed_seed(ess_df):
    """(d) Identical seed ⇒ byte-identical cohort."""
    a = synthesize_placebo_personas(ess_df, n=N, seed=SEED)
    b = synthesize_placebo_personas(ess_df, n=N, seed=SEED)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]


def test_different_seed_changes_pairing(ess_df):
    """Different seed ⇒ different trait pairing (but same marginals)."""
    a = synthesize_placebo_personas(ess_df, n=N, seed=1)
    b = synthesize_placebo_personas(ess_df, n=N, seed=2)
    assert [r.model_dump() for r in a] != [r.model_dump() for r in b]


def test_records_build_valid_agents(placebo):
    """(e) Placebo records pass AgentProfile.__post_init__ via the pipeline."""
    agents = persona_records_to_agents(placebo, policy=MockPolicy(), memory_size=10)
    assert len(agents) == N
    for ag in agents:
        tp = ag.profile.trust_people
        assert tp is None or 0.0 <= tp <= 1.0
        assert ag.profile.gender in (1, 2, None)


def test_small_cohort_no_crash(ess_df):
    """<2 personas: scramble is a no-op, no exception."""
    one = synthesize_placebo_personas(ess_df, n=1, seed=SEED)
    assert len(one) == 1
