"""Regenerate benchmark/LEADERBOARD.md from benchmark/submissions/*.json.

The leaderboard is a thin presentation layer over the score-card JSON
files. Submissions are sorted within tier by ``delta_B_RLHF_relative``
ascending (more negative = stronger bias reduction = better).

Run with::

    python benchmark/build_leaderboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "benchmark" / "submissions"
OUT_PATH = REPO_ROOT / "benchmark" / "LEADERBOARD.md"

TIER_ORDER = ("confirmatory", "extended", "standard", "pilot")


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _row(card: dict) -> dict:
    sub = card["submission"]
    sc = card["scores"]
    return {
        "name": sub["name"],
        "team": sub["team"],
        "tier": sub["tier"],
        "model": sub["llm"]["model_id"],
        "rag": ",".join(sub["grounding_config"]["rag"]),
        "memory": sub["grounding_config"]["memory"],
        "b_rlhf_a": sc["condition_A"]["B_RLHF"],
        "b_rlhf_b": sc["condition_B"]["B_RLHF"],
        "brm_a": sc["condition_A"]["BRM_composite"],
        "brm_b": sc["condition_B"]["BRM_composite"],
        "delta_b_rlhf": sc["delta_B_RLHF_relative"],
        "delta_brm": sc["delta_BRM_composite"],
        "schema_version": card.get("schema_version", "?"),
        "date": sub.get("date", "?"),
    }


def _render(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# B_RLHF Benchmark Leaderboard\n")
    lines.append(
        "Auto-generated from `benchmark/submissions/*.json`. To regenerate "
        "run `python benchmark/build_leaderboard.py`. The protocol that "
        "submissions must follow is `benchmark/SPECIFICATION.md` (v1.0).\n"
    )
    lines.append(
        "Sort order: by tier (confirmatory → pilot), then by "
        "`delta_B_RLHF_relative` ascending (more negative = stronger "
        "bias reduction under grounding).\n"
    )

    if not rows:
        lines.append("_No submissions on disk yet._\n")
        return "\n".join(lines)

    for tier in TIER_ORDER:
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        tier_rows.sort(key=lambda r: r["delta_b_rlhf"])
        lines.append(f"\n## Tier: {tier}\n")
        lines.append(
            "| Submission | Team | Model | Memory | RAG | B_RLHF (A→B) | BRM (A→B) | ΔB_RLHF | ΔBRM | Schema | Date |"
        )
        lines.append("|---|---|---|---|---|---|---|---:|---:|---|---|")
        for r in tier_rows:
            lines.append(
                f"| `{r['name']}` | {r['team']} | {r['model']} | "
                f"{r['memory']} | {r['rag']} | "
                f"{_fmt(r['b_rlhf_a'])} → {_fmt(r['b_rlhf_b'])} | "
                f"{_fmt(r['brm_a'])} → {_fmt(r['brm_b'])} | "
                f"{r['delta_b_rlhf']:+.3f} | {r['delta_brm']:+.3f} | "
                f"{r['schema_version']} | {r['date']} |"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    rows: list[dict] = []
    if SUBMISSIONS_DIR.exists():
        for path in sorted(SUBMISSIONS_DIR.glob("*.json")):
            try:
                rows.append(_row(json.loads(path.read_text())))
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"warning: skipping {path.name} ({exc})")

    OUT_PATH.write_text(_render(rows))
    print(f"Leaderboard written to {OUT_PATH} ({len(rows)} submission(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
