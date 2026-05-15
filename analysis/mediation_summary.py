"""Aggregate 2×2 factorial mediation cells into analysis/tables/mediation.json.

Paper §7.3 reports preliminary persona/RAG/interaction shares. This
script converts that prose into an auditable JSON artefact by reading
the corresponding factorial cells from ``experiments/`` and applying
the canonical decomposition in ``metrics/mediation.py``.

Required experiment cells (one per seed in ``SEEDS``)::

    baseline_s{seed}        — no persona, no RAG     (Condition A)
    persona_only_s{seed}    — persona, no RAG
    rag_only_s{seed}        — RAG, no persona
    full_grounded_s{seed}   — persona + RAG          (Condition B)

If any cell is missing the script emits a structured "cells_missing"
JSON listing exactly what needs to be run. This converts the §7.3
hidden gap into an audit-visible one rather than silently faking numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from benchmark.score import score_experiment
from metrics.mediation import compute_mediation_decomposition

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / "experiments"
OUT_PATH = REPO_ROOT / "analysis" / "tables" / "mediation.json"

SEEDS = (42, 43, 44)

# Canonical cell name → list of acceptable experiment-directory prefixes.
# The script picks the first match per seed. Adjust this map when the
# factorial runner lands.
CELL_PREFIX_CANDIDATES: dict[str, tuple[str, ...]] = {
    "baseline": ("pure_llm_ess_persona_s",),
    "persona_only": ("ablation_persona_only_s", "ablation_rich_persona_s"),
    "rag_only": ("ablation_rag_only_s",),
    "full_grounded": ("grounded_llm_ess_persona_s",),
}


def _resolve_cell(cell: str, seed: int) -> Path | None:
    for prefix in CELL_PREFIX_CANDIDATES[cell]:
        path = EXP_ROOT / f"{prefix}{seed}"
        if path.exists() and (path / "events.jsonl").exists():
            return path
    return None


def _emit_missing_cells_stub(missing: dict[int, list[str]]) -> dict:
    """Write a structured stub when factorial cells are unavailable."""
    needed = sorted({f"{cell}_s{seed}" for seed, cells in missing.items() for cell in cells})
    return {
        "_status": "cells_missing",
        "_note": (
            "The 2×2 factorial mediation cells required by paper §7.3 are "
            "not all present on disk. Run the missing experiments below, "
            "then re-execute `python analysis/mediation_summary.py`."
        ),
        "missing_cells": needed,
        "candidate_prefixes": {cell: list(pref) for cell, pref in CELL_PREFIX_CANDIDATES.items()},
        "expected_seeds": list(SEEDS),
    }


def main() -> int:
    per_seed_metrics: dict[int, dict[str, dict]] = {}
    missing: dict[int, list[str]] = {}
    for seed in SEEDS:
        cells: dict[str, dict] = {}
        for cell in CELL_PREFIX_CANDIDATES:
            path = _resolve_cell(cell, seed)
            if path is None:
                missing.setdefault(seed, []).append(cell)
                continue
            cells[cell] = score_experiment(path)
        if not missing.get(seed):
            per_seed_metrics[seed] = cells

    if missing:
        stub = _emit_missing_cells_stub(missing)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(stub, indent=2) + "\n")
        print(f"Wrote cells-missing stub to {OUT_PATH}")
        for seed, cells in missing.items():
            print(f"  seed {seed}: missing {cells}")
        return 0

    # Decompose per seed, then pool means across seeds.
    decompositions: dict[int, dict[str, dict[str, float]]] = {}
    for seed, cells in per_seed_metrics.items():
        decompositions[seed] = {}
        for metric in ("coop_rate", "BRM_composite", "B_RLHF", "final_gini"):
            decompositions[seed][metric] = compute_mediation_decomposition(
                full_grounded_coop=cells["full_grounded"][metric],
                persona_only_coop=cells["persona_only"][metric],
                rag_only_coop=cells["rag_only"][metric],
                baseline_coop=cells["baseline"][metric],
            )

    pooled: dict[str, dict[str, float]] = {}
    for metric in next(iter(decompositions.values())).keys():
        pooled[metric] = {}
        keys = next(iter(decompositions.values()))[metric].keys()
        for k in keys:
            pooled[metric][k] = round(mean(decompositions[seed][metric][k] for seed in SEEDS), 4)

    out = {
        "_status": "complete",
        "_note": "2x2 factorial mediation; values pooled (mean) across seeds.",
        "seeds": list(SEEDS),
        "pooled_decomposition": pooled,
        "per_seed_decomposition": {
            str(seed): {m: {k: round(v, 4) for k, v in d.items()} for m, d in decomps.items()}
            for seed, decomps in decompositions.items()
        },
        "per_seed_raw_metrics": {
            str(seed): {
                cell: {k: cells[cell][k] for k in ("coop_rate", "BRM_composite", "B_RLHF", "final_gini")}
                for cell in cells
            }
            for seed, cells in per_seed_metrics.items()
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Mediation summary written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
