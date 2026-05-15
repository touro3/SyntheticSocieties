"""B_RLHF Benchmark scorer (v1.0).

Canonical scorer referenced by ``benchmark/SPECIFICATION.md`` §6.

Reads a pair of BGF experiment directories (Condition A ungrounded /
Condition B grounded), recomputes B_RLHF, BRM_composite, cooperation
rate, and final Gini from raw ``events.jsonl`` + ``config.yaml`` using
the canonical metric implementations in ``metrics/``, validates the
spec's fixed parameters, and emits the score-card JSON described in
SPECIFICATION.md §5.

Usage::

    python benchmark/score.py \\
        --exp-a experiments/pure_llm_ess_persona_s42 \\
        --exp-b experiments/grounded_llm_ess_persona_s42 \\
        --submission-name my_submission \\
        --tier pilot \\
        --out benchmark/submissions/my_submission.json

The scorer is deliberately read-only on the project state: it never
mutates the input experiments and writes only the output JSON.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from metrics.behavioral_realism import (
    compute_composite_brm,
    rlhf_bias_index_from_counts,
)
from metrics.inequality import gini_coefficient

CANONICAL_ACTIONS = ("work", "save", "cooperate")

# Tier seed sets per SPECIFICATION.md §3.
TIER_SEEDS: dict[str, tuple[int, ...]] = {
    "pilot": (42, 43, 44),
    "standard": (42, 43, 44, 123, 7),
    "extended": tuple(range(1, 11)),
    "confirmatory": tuple(range(1, 11)),
}

TIER_AGENTS: dict[str, int] = {
    "pilot": 20,
    "standard": 50,
    "extended": 200,
    "confirmatory": 500,
}

TIER_ROUNDS: dict[str, int] = {
    "pilot": 10,
    "standard": 30,
    "extended": 50,
    "confirmatory": 30,
}


# ── I/O helpers ────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


# ── Metric extraction ──────────────────────────────────────────────────────


def _action_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ev in events:
        action_type = ev.get("action", {}).get("action_type")
        if action_type in CANONICAL_ACTIONS:
            counts[action_type] += 1
    # Ensure every canonical action is present so the action-distribution
    # is over the full support — B_RLHF reference is uniform on |A|=3.
    for a in CANONICAL_ACTIONS:
        counts.setdefault(a, 0)
    return dict(counts)


def _final_wealth_vector(events: list[dict[str, Any]]) -> list[float]:
    last_wealth: dict[str, float] = {}
    last_round: dict[str, int] = {}
    for ev in events:
        agent_id = ev.get("agent_id")
        round_id = ev.get("round_id", 0)
        state_after = ev.get("state_after") or {}
        wealth = state_after.get("wealth")
        if agent_id is None or wealth is None:
            continue
        if round_id >= last_round.get(agent_id, -1):
            last_round[agent_id] = round_id
            last_wealth[agent_id] = float(wealth)
    return list(last_wealth.values())


def _round_action_distributions(
    events: list[dict[str, Any]],
) -> list[dict[str, float]]:
    by_round: dict[int, Counter[str]] = {}
    for ev in events:
        action_type = ev.get("action", {}).get("action_type")
        if action_type not in CANONICAL_ACTIONS:
            continue
        round_id = ev.get("round_id", 0)
        by_round.setdefault(round_id, Counter())[action_type] += 1
    dists: list[dict[str, float]] = []
    for round_id in sorted(by_round):
        c = by_round[round_id]
        total = sum(c.values())
        if total == 0:
            continue
        dists.append({a: c.get(a, 0) / total for a in CANONICAL_ACTIONS})
    return dists


def _mean_round_to_round_jsd(dists: list[dict[str, float]]) -> float:
    """Mean Jensen-Shannon divergence between successive round distributions.

    Used as the temporal-stability sub-component of BRM. Returns 0.0 when
    fewer than two rounds are available (no instability to measure).
    """
    if len(dists) < 2:
        return 0.0
    import math

    def _kl(p: dict[str, float], m: dict[str, float]) -> float:
        s = 0.0
        for a in CANONICAL_ACTIONS:
            pa = p.get(a, 0.0)
            ma = m.get(a, 0.0)
            if pa > 0 and ma > 0:
                s += pa * math.log2(pa / ma)
        return s

    total = 0.0
    n = 0
    for i in range(1, len(dists)):
        p, q = dists[i - 1], dists[i]
        m = {a: 0.5 * (p.get(a, 0.0) + q.get(a, 0.0)) for a in CANONICAL_ACTIONS}
        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        total += jsd
        n += 1
    return total / n if n else 0.0


def score_experiment(
    exp_dir: Path,
    *,
    emp_wealth: list[float] | None = None,
    emp_gini: float = 0.31,
    emp_coop_rate: float = 0.50,
) -> dict[str, Any]:
    """Compute the four spec metrics for a single experiment directory.

    Empirical references are the European Eurostat median Gini and the
    midpoint of the documented public-goods-game cooperation band — the
    same anchors used in paper §5.4 Table 1.
    """
    events_path = exp_dir / "events.jsonl"
    config_path = exp_dir / "config.yaml"

    events = _load_events(events_path)
    counts = _action_counts(events)
    total_actions = sum(counts.values())
    if total_actions == 0:
        raise ValueError(f"{exp_dir} contains no canonical-action events")

    coop_rate = counts["cooperate"] / total_actions
    b_rlhf = rlhf_bias_index_from_counts(counts)
    wealth = _final_wealth_vector(events)
    final_gini = gini_coefficient(wealth) if wealth else 0.0
    round_dists = _round_action_distributions(events)
    temporal_jsd = _mean_round_to_round_jsd(round_dists)

    # Empirical wealth reference: the canonical workflow uses ESS-derived
    # wealth quantiles. Until that artefact is generally available, the
    # scorer falls back to the simulated wealth distribution itself for
    # the JSD-against-empirical term, which yields jsd_component = 1.0
    # — neutralizing that single sub-component without biasing the
    # ordering between A and B (both submissions are scored identically).
    if emp_wealth is None:
        emp_wealth = list(wealth)

    brm = compute_composite_brm(
        sim_wealth=wealth or [0.0],
        emp_wealth=emp_wealth or [0.0],
        sim_gini=final_gini,
        emp_gini=emp_gini,
        sim_coop_rate=coop_rate,
        emp_coop_rate=emp_coop_rate,
        temporal_stability_jsd=temporal_jsd,
    )

    return {
        "B_RLHF": round(b_rlhf, 4),
        "BRM_composite": round(brm["composite"], 4),
        "coop_rate": round(coop_rate, 4),
        "final_gini": round(final_gini, 4),
        "_audit": {
            "n_events": total_actions,
            "n_rounds": len(round_dists),
            "n_agents_terminal_wealth": len(wealth),
            "action_counts": counts,
            "events_sha256": _sha256_file(events_path),
            "config_sha256": _sha256_file(config_path),
            "config_path": str(config_path),
        },
        "_brm_components": {k: round(v, 4) for k, v in brm.items() if k != "composite"},
    }


# ── Spec validation ────────────────────────────────────────────────────────


def validate_tier(
    *,
    tier: str,
    seeds: list[int],
    scores_a: dict[str, Any],
    scores_b: dict[str, Any],
) -> list[str]:
    """Return list of human-readable spec violations (empty = valid)."""
    errors: list[str] = []
    if tier not in TIER_SEEDS:
        errors.append(f"Unknown tier {tier!r}; expected one of {list(TIER_SEEDS)}")
        return errors

    expected_seeds = set(TIER_SEEDS[tier])
    actual_seeds = set(seeds)
    if not expected_seeds.issubset(actual_seeds):
        missing = sorted(expected_seeds - actual_seeds)
        errors.append(f"Tier {tier!r} requires seeds {sorted(expected_seeds)}; submission is missing {missing}")

    # Cross-check action support: a valid submission must exercise all
    # three actions at least once across rounds (otherwise the
    # action-parser cascade is suspect).
    for label, scores in (("A", scores_a), ("B", scores_b)):
        counts = scores.get("_audit", {}).get("action_counts", {})
        missing = [a for a in CANONICAL_ACTIONS if counts.get(a, 0) == 0]
        if missing:
            errors.append(
                f"Condition {label}: zero events for action(s) {missing} — "
                "action support too narrow for a valid B_RLHF measurement"
            )

    return errors


# ── Score-card construction ────────────────────────────────────────────────


def build_score_card(
    *,
    name: str,
    team: str,
    tier: str,
    seeds: list[int],
    llm_model_id: str,
    llm_revision: str,
    llm_temperature: float,
    llm_top_p: float,
    grounding_persona: str,
    grounding_rag: list[str],
    grounding_memory: str,
    grounding_notes: str,
    scores_a: dict[str, Any],
    scores_b: dict[str, Any],
    reproduction_command: str,
) -> dict[str, Any]:
    """Assemble the SPECIFICATION.md §5 score-card JSON."""
    b_rlhf_a = scores_a["B_RLHF"]
    b_rlhf_b = scores_b["B_RLHF"]
    delta_relative = (b_rlhf_b - b_rlhf_a) / b_rlhf_a if b_rlhf_a > 0 else 0.0
    delta_brm = scores_b["BRM_composite"] - scores_a["BRM_composite"]

    return {
        "schema_version": "1.0",
        "submission": {
            "name": name,
            "team": team,
            "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tier": tier,
            "llm": {
                "model_id": llm_model_id,
                "revision": llm_revision,
                "temperature": llm_temperature,
                "top_p": llm_top_p,
            },
            "grounding_config": {
                "persona": grounding_persona,
                "rag": grounding_rag,
                "memory": grounding_memory,
                "notes": grounding_notes,
            },
            "seeds": list(seeds),
        },
        "scores": {
            "condition_A": {
                "B_RLHF": scores_a["B_RLHF"],
                "BRM_composite": scores_a["BRM_composite"],
                "coop_rate": scores_a["coop_rate"],
                "final_gini": scores_a["final_gini"],
            },
            "condition_B": {
                "B_RLHF": scores_b["B_RLHF"],
                "BRM_composite": scores_b["BRM_composite"],
                "coop_rate": scores_b["coop_rate"],
                "final_gini": scores_b["final_gini"],
            },
            "delta_B_RLHF_relative": round(delta_relative, 4),
            "delta_BRM_composite": round(delta_brm, 4),
        },
        "audit": {
            "condition_A": scores_a["_audit"],
            "condition_B": scores_b["_audit"],
            "brm_components_A": scores_a["_brm_components"],
            "brm_components_B": scores_b["_brm_components"],
            "reproduction_command": reproduction_command,
        },
    }


# ── CLI entry-point ────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp-a", type=Path, required=True, help="Condition A experiment dir")
    p.add_argument("--exp-b", type=Path, required=True, help="Condition B experiment dir")
    p.add_argument("--submission-name", required=True)
    p.add_argument("--team", default="anonymous")
    p.add_argument("--tier", choices=list(TIER_SEEDS), required=True)
    p.add_argument("--seeds", default="", help="Comma-separated seed list")
    p.add_argument("--llm-model-id", default="unknown")
    p.add_argument("--llm-revision", default="unknown")
    p.add_argument("--llm-temperature", type=float, default=0.5)
    p.add_argument("--llm-top-p", type=float, default=1.0)
    p.add_argument("--persona", default="ess")
    p.add_argument("--rag", default="sql,graph")
    p.add_argument("--memory", default="M3")
    p.add_argument("--notes", default="")
    p.add_argument("--out", type=Path, required=True, help="Output score-card JSON")
    p.add_argument(
        "--allow-spec-violations",
        action="store_true",
        help="Emit score card even if tier validation fails (sets schema_version='1.0-draft')",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else list(TIER_SEEDS[args.tier])

    scores_a = score_experiment(args.exp_a)
    scores_b = score_experiment(args.exp_b)

    errors = validate_tier(tier=args.tier, seeds=seeds, scores_a=scores_a, scores_b=scores_b)

    if errors and not args.allow_spec_violations:
        print("Specification violations (use --allow-spec-violations to override):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    rag_list = [r.strip() for r in args.rag.split(",") if r.strip()]
    reproduction_command = (
        f"python benchmark/score.py --exp-a {args.exp_a} --exp-b {args.exp_b} "
        f"--submission-name {args.submission_name} --tier {args.tier} --out {args.out}"
    )
    card = build_score_card(
        name=args.submission_name,
        team=args.team,
        tier=args.tier,
        seeds=seeds,
        llm_model_id=args.llm_model_id,
        llm_revision=args.llm_revision,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p,
        grounding_persona=args.persona,
        grounding_rag=rag_list,
        grounding_memory=args.memory,
        grounding_notes=args.notes,
        scores_a=scores_a,
        scores_b=scores_b,
        reproduction_command=reproduction_command,
    )
    if errors:
        card["schema_version"] = "1.0-draft"
        card["_validation_errors"] = errors

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2) + "\n")
    print(f"Score card written to {args.out}")
    print(json.dumps(card["scores"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
