#!/usr/bin/env python3
"""Build the Condition D (Rule-Based ESS) Table-3b row — sub-phase 28.2.

Condition D is the deterministic, non-LLM ESS baseline (no GPU). This reads
the on-disk Condition-D experiment dirs, computes per-seed cooperation rate,
wealth Gini and B_RLHF, and aggregates each across seeds with the same shared
BCa 95% CI used by the ten-seed confirmatory pipeline.

Two scales are reported per Decision D-0 ("run both"):
  * N=500, seeds 42,123,7  (prefix cmp_rbe_s)   — paper headline anchor
  * N=100, seeds 1,2,3     (prefix cond_d_rbe_s) — robustness footnote

Press-play: if no experiment dirs exist yet it writes status=awaiting_runs
and exits 0, so it is safe to wire into CI.

Usage:
    python scripts/build_condition_d_table.py                       # N=500
    python scripts/build_condition_d_table.py \\
        --seeds 1,2,3 --prefix cond_d_rbe_s \\
        --out-json analysis/condition_d_results_n100.json            # N=100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from metrics.behavioral_realism import rlhf_bias_index_from_counts  # noqa: E402
from metrics.statistical_inference import bca_ci  # noqa: E402

_ACTIONS = ("work", "save", "cooperate")


def load_condition_d_run(run_dir: Path) -> dict | None:
    """Per-run coop_rate / gini / b_rlhf + scale, or None if the dir is
    missing or too sparse to score."""
    run_dir = Path(run_dir)
    summary_p = run_dir / "summary.json"
    meta_p = run_dir / "metadata.json"
    if not summary_p.exists():
        return None
    summary = json.loads(summary_p.read_text())
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

    eac = {a: int((summary.get("event_action_counts") or {}).get(a, 0)) for a in _ACTIONS}
    total = sum(eac.values())
    if total <= 0:
        return None
    gini = (summary.get("wealth") or {}).get("gini")
    if gini is None:
        return None
    return {
        "coop_rate": eac["cooperate"] / total,
        "gini": float(gini),
        "b_rlhf": float(rlhf_bias_index_from_counts(eac)),
        "population_size": meta.get("population_size"),
        "rounds": meta.get("rounds"),
    }


def _fmt(stat: dict) -> str:
    lo, hi = stat["ci95"]
    return f"{stat['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def build(
    seeds: list[int],
    prefix: str = "cmp_rbe_s",
    experiments_dir: str | Path = "experiments",
    out_json: str | Path = "analysis/condition_d_results.json",
) -> tuple[dict, str]:
    """Aggregate Condition-D runs into a results dict + a Markdown Table-3b row.

    Raises ValueError if found runs disagree on (N, T) — a hard guard against
    silently averaging incomparable scales into one fabricated row.
    """
    experiments_dir = Path(experiments_dir)
    runs: dict[int, dict] = {}
    for s in seeds:
        m = load_condition_d_run(experiments_dir / f"{prefix}{s}")
        if m is None:
            print(f"  [skip] {prefix}{s}: missing or unscorable", file=sys.stderr)
            continue
        runs[s] = m

    out_json = Path(out_json)
    if not runs:
        payload = {
            "status": "awaiting_runs",
            "message": (
                f"No Condition-D runs found for prefix '{prefix}' seeds {seeds}. "
                f"Launch scripts/launch_cond_d.sh (N=500) or "
                f"`run_full_pipeline.py --condition D` (N=100), then re-run."
            ),
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        return payload, ""

    pops = {m["population_size"] for m in runs.values()}
    rnds = {m["rounds"] for m in runs.values()}
    if len(pops) > 1 or len(rnds) > 1:
        raise ValueError(
            f"Mixed Condition-D scale across seeds (N={sorted(pops)}, T={sorted(rnds)}); "
            "refusing to aggregate incomparable runs into one row."
        )

    found = sorted(runs)
    per_metric: dict[str, dict] = {}
    for metric in ("coop_rate", "gini", "b_rlhf"):
        vals = np.array([runs[s][metric] for s in found], dtype=float)
        lo, hi = bca_ci(vals)
        per_metric[metric] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "ci95": [lo, hi],
            "per_seed": {str(s): float(runs[s][metric]) for s in found},
        }

    results = {
        "status": "complete",
        "experiment": {
            "condition": "D",
            "policy": "rule_based_ess",
            "population_size": pops.pop(),
            "rounds": rnds.pop(),
            "seeds_requested": seeds,
            "seeds_found": found,
            "source_dirs": [f"{prefix}{s}" for s in found],
        },
        "per_metric": per_metric,
        "note": (
            "Deterministic non-LLM ESS baseline (no GPU). Identical action "
            "counts across seeds collapse coop_rate/B_RLHF CIs to a point "
            "(expected for a deterministic policy); only Gini varies with the "
            "ESS population draw. Not directly comparable to the cross-model "
            "Table 3 (different N/T, no within-D A/B baseline)."
        ),
    }
    md_row = (
        f"| Rule-Based ESS | D | {_fmt(per_metric['coop_rate'])} "
        f"| {_fmt(per_metric['gini'])} | {_fmt(per_metric['b_rlhf'])} | — |"
    )
    results["markdown_row"] = md_row

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    return results, md_row


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Condition D Table-3b row")
    ap.add_argument("--seeds", default="42,123,7", help="comma-separated seeds")
    ap.add_argument("--prefix", default="cmp_rbe_s", help="experiment-id prefix")
    ap.add_argument("--experiments-dir", default="experiments")
    ap.add_argument("--out-json", default="analysis/condition_d_results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    results, md_row = build(seeds, args.prefix, args.experiments_dir, args.out_json)
    print(json.dumps(results, indent=2))
    if md_row:
        print("\nTable-3b row:\n" + md_row)
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
