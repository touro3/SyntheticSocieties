"""Build the BGF reference submission for the B_RLHF leaderboard.

Aggregates the three pilot seeds (42, 43, 44) of the
``pure_llm_ess_persona`` (Condition A) and ``grounded_llm_ess_persona``
(Condition B) experiment families on disk into a single pilot-tier
score card, written to ``benchmark/submissions/bgf_paper_pilot.json``.

Per-seed scores are produced by ``benchmark.score.score_experiment`` and
pooled (mean) into the headline ``condition_A`` / ``condition_B``
metrics, preserving the per-seed audit trail in ``audit.per_seed``.

Run with::

    python benchmark/build_bgf_reference.py

The script is idempotent and re-runnable: the output file is fully
regenerated each invocation.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from statistics import mean

from benchmark.score import (
    TIER_SEEDS,
    build_score_card,
    score_experiment,
    validate_tier,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / "experiments"
OUT_PATH = REPO_ROOT / "benchmark" / "submissions" / "bgf_paper_pilot.json"

TIER = "pilot"
SEEDS = list(TIER_SEEDS[TIER])


def _pool(scores: list[dict]) -> dict:
    keys = ("B_RLHF", "BRM_composite", "coop_rate", "final_gini")
    return {k: round(mean(s[k] for s in scores), 4) for k in keys}


def main() -> int:
    per_seed_a: list[dict] = []
    per_seed_b: list[dict] = []
    for seed in SEEDS:
        exp_a = EXP_ROOT / f"pure_llm_ess_persona_s{seed}"
        exp_b = EXP_ROOT / f"grounded_llm_ess_persona_s{seed}"
        if not exp_a.exists() or not exp_b.exists():
            raise FileNotFoundError(f"Missing experiment dir for seed {seed}: A={exp_a.exists()}, B={exp_b.exists()}")
        sa = score_experiment(exp_a)
        sb = score_experiment(exp_b)
        per_seed_a.append({"seed": seed, **sa})
        per_seed_b.append({"seed": seed, **sb})

    pooled_a = _pool(per_seed_a)
    pooled_b = _pool(per_seed_b)
    pooled_a_card_input = {
        **pooled_a,
        "_audit": {
            "per_seed": [{k: v for k, v in s.items() if k != "_brm_components"} for s in per_seed_a],
            "action_counts": {
                a: sum(s["_audit"]["action_counts"].get(a, 0) for s in per_seed_a)
                for a in ("work", "save", "cooperate")
            },
        },
        "_brm_components": {
            k: round(mean(s["_brm_components"][k] for s in per_seed_a), 4) for k in per_seed_a[0]["_brm_components"]
        },
    }
    pooled_b_card_input = {
        **pooled_b,
        "_audit": {
            "per_seed": [{k: v for k, v in s.items() if k != "_brm_components"} for s in per_seed_b],
            "action_counts": {
                a: sum(s["_audit"]["action_counts"].get(a, 0) for s in per_seed_b)
                for a in ("work", "save", "cooperate")
            },
        },
        "_brm_components": {
            k: round(mean(s["_brm_components"][k] for s in per_seed_b), 4) for k in per_seed_b[0]["_brm_components"]
        },
    }

    errors = validate_tier(
        tier=TIER,
        seeds=SEEDS,
        scores_a=pooled_a_card_input,
        scores_b=pooled_b_card_input,
    )

    card = build_score_card(
        name="bgf_paper_pilot",
        team="BGF authors (reference)",
        tier=TIER,
        seeds=SEEDS,
        llm_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        llm_revision="hf-revision-pinned-in-config",
        llm_temperature=0.5,
        llm_top_p=1.0,
        grounding_persona="ess",
        grounding_rag=["sql", "graph"],
        grounding_memory="M3",
        grounding_notes=(
            "Reference submission constructed from the canonical pilot "
            "experiments pure_llm_ess_persona_s{42,43,44} (Condition A) "
            "and grounded_llm_ess_persona_s{42,43,44} (Condition B). "
            "Per-seed audit retained in audit.condition_*.per_seed."
        ),
        scores_a=pooled_a_card_input,
        scores_b=pooled_b_card_input,
        reproduction_command="python benchmark/build_bgf_reference.py",
    )
    if errors:
        card["schema_version"] = "1.0-draft"
        card["_validation_errors"] = errors

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(card, indent=2) + "\n")
    print(f"Reference submission written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
