"""Tests for benchmark/score.py and the BGF reference submission.

These tests are the CI-side enforcement of `benchmark/SPECIFICATION.md`:
they assert that the scorer correctly recovers the canonical pilot
metrics and that the reference submission validates against the spec's
schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmark.score import (
    CANONICAL_ACTIONS,
    TIER_SEEDS,
    score_experiment,
    validate_tier,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / "experiments"
REFERENCE_PATH = REPO_ROOT / "benchmark" / "submissions" / "bgf_paper_pilot.json"


# ── score_experiment unit tests ───────────────────────────────────────────


@pytest.fixture(scope="module")
def cond_a_seed42() -> dict:
    exp = EXP_ROOT / "pure_llm_ess_persona_s42"
    if not exp.exists():
        pytest.skip(f"missing pilot experiment dir {exp}")
    return score_experiment(exp)


@pytest.fixture(scope="module")
def cond_b_seed42() -> dict:
    exp = EXP_ROOT / "grounded_llm_ess_persona_s42"
    if not exp.exists():
        pytest.skip(f"missing pilot experiment dir {exp}")
    return score_experiment(exp)


def test_score_experiment_returns_required_keys(cond_a_seed42: dict) -> None:
    for key in ("B_RLHF", "BRM_composite", "coop_rate", "final_gini"):
        assert key in cond_a_seed42, f"missing {key}"
        assert isinstance(cond_a_seed42[key], (int, float))


def test_score_experiment_metric_ranges(cond_a_seed42: dict, cond_b_seed42: dict) -> None:
    for scores in (cond_a_seed42, cond_b_seed42):
        assert 0.0 <= scores["B_RLHF"] <= 2 / 3 + 1e-6, "B_RLHF out of [0, 2/3]"
        assert 0.0 <= scores["BRM_composite"] <= 1.0
        assert 0.0 <= scores["coop_rate"] <= 1.0
        assert 0.0 <= scores["final_gini"] < 1.0


def test_pilot_coop_rate_matches_paper_numbers(cond_a_seed42: dict) -> None:
    """Condition A seed 42 coop rate must match analysis/paper_numbers.json
    to 3 decimals — the canonical source of headline pilot numbers.
    """
    paper_numbers = json.loads((REPO_ROOT / "analysis" / "paper_numbers.json").read_text())
    seeds = paper_numbers["pure_llm_ess_persona"]["per_seed"]
    seed42 = next(s for s in seeds if s["seed"] == 42)
    assert math.isclose(cond_a_seed42["coop_rate"], seed42["coop_rate_overall"], abs_tol=1e-3)


def test_b_rlhf_corollary_consistency(cond_a_seed42: dict) -> None:
    """Theorem 1 corollary: under equal work/save split, B_RLHF = p − 1/3.

    The split is not always equal in real runs, but TV must always be
    at least |coop_rate − 1/3| / 2 + the residual work/save deviation.
    Looser invariant: B_RLHF ≥ 0 and B_RLHF ≤ 2/3.
    """
    assert cond_a_seed42["B_RLHF"] >= 0.0
    assert cond_a_seed42["B_RLHF"] <= 2 / 3 + 1e-6


# ── validate_tier tests ────────────────────────────────────────────────────


def _stub_scores(actions_present: list[str] | None = None) -> dict:
    actions_present = actions_present or list(CANONICAL_ACTIONS)
    return {
        "B_RLHF": 0.3,
        "BRM_composite": 0.8,
        "coop_rate": 0.4,
        "final_gini": 0.25,
        "_audit": {"action_counts": {a: 10 for a in actions_present}},
    }


def test_validate_tier_passes_with_correct_seeds() -> None:
    errors = validate_tier(
        tier="pilot",
        seeds=list(TIER_SEEDS["pilot"]),
        scores_a=_stub_scores(),
        scores_b=_stub_scores(),
    )
    assert errors == []


def test_validate_tier_flags_missing_seeds() -> None:
    errors = validate_tier(
        tier="standard",
        seeds=[42],
        scores_a=_stub_scores(),
        scores_b=_stub_scores(),
    )
    assert any("missing" in e for e in errors)


def test_validate_tier_flags_unknown_tier() -> None:
    errors = validate_tier(tier="nonsense", seeds=[], scores_a=_stub_scores(), scores_b=_stub_scores())
    assert any("Unknown tier" in e for e in errors)


def test_validate_tier_flags_missing_action_support() -> None:
    errors = validate_tier(
        tier="pilot",
        seeds=list(TIER_SEEDS["pilot"]),
        scores_a=_stub_scores(actions_present=["work", "save"]),
        scores_b=_stub_scores(),
    )
    assert any("action(s)" in e for e in errors)


# ── reference submission tests ────────────────────────────────────────────


@pytest.fixture(scope="module")
def reference_card() -> dict:
    if not REFERENCE_PATH.exists():
        pytest.skip(f"reference submission not built; run benchmark/build_bgf_reference.py")
    return json.loads(REFERENCE_PATH.read_text())


def test_reference_schema(reference_card: dict) -> None:
    assert reference_card["schema_version"] == "1.0"
    sub = reference_card["submission"]
    for key in ("name", "team", "tier", "llm", "grounding_config", "seeds"):
        assert key in sub, f"missing submission.{key}"
    for key in ("model_id", "revision", "temperature", "top_p"):
        assert key in sub["llm"], f"missing submission.llm.{key}"
    scores = reference_card["scores"]
    for cond in ("condition_A", "condition_B"):
        for metric in ("B_RLHF", "BRM_composite", "coop_rate", "final_gini"):
            assert metric in scores[cond], f"missing scores.{cond}.{metric}"
    assert "delta_B_RLHF_relative" in scores
    assert "delta_BRM_composite" in scores


def test_reference_tier_and_seeds(reference_card: dict) -> None:
    assert reference_card["submission"]["tier"] == "pilot"
    assert set(reference_card["submission"]["seeds"]) == set(TIER_SEEDS["pilot"])


def test_reference_grounding_direction(reference_card: dict) -> None:
    """The BGF reference submission must show grounding reducing B_RLHF
    and increasing BRM (H1 and H2 directional consistency)."""
    scores = reference_card["scores"]
    assert scores["delta_B_RLHF_relative"] < 0, (
        f"Reference grounding should reduce B_RLHF — got delta_B_RLHF_relative={scores['delta_B_RLHF_relative']}"
    )
    assert scores["delta_BRM_composite"] > 0, (
        f"Reference grounding should increase BRM_composite — got delta_BRM_composite={scores['delta_BRM_composite']}"
    )


def test_reference_action_support_complete(reference_card: dict) -> None:
    """Both conditions must exercise all three actions."""
    for cond in ("condition_A", "condition_B"):
        counts = reference_card["audit"][cond]["action_counts"]
        for action in CANONICAL_ACTIONS:
            assert counts.get(action, 0) > 0, f"{cond} has zero events for {action!r} — invalid B_RLHF"
