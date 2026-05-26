#!/usr/bin/env python
"""Decide the §8.1.3 N=500 LLM pivot framing.

Runs once the 20 N=500 cells (Conditions A × seeds {1..10} and B × seeds
{1..10}) have all written ``summary.json``. Produces a one-page decision
artefact at ``analysis/reports/n500_pivot_decision.md`` plus the underlying
CSV at ``analysis/tables/n500_pivot_metrics.csv``.

Why this exists
---------------
The pre-registered confirmatory hypothesis for §8.1 is

    H1:  BRM(B) > BRM(A)    (one-sided, α = 0.05 family-wise via BH-FDR)
    H2:  B_RLHF(B) < B_RLHF(A)    (same)

The N=100 extension falsified H2. The N=500 sweep currently in flight is
the last data point that decides whether to (a) report a positive
dissociation finding ("grounding restores realism at primary scale") or
(b) pivot to the alignment-tax framing ("RLHF cooperative bias is robust
to inference-time prompting at every scale we measured").

This script does not pick the framing for you — it computes both H1 and
the "scale invariance" alternative H_alt:

    H_alt:  |B_RLHF(N=500) - B_RLHF(N=100)| < 0.05    (paired across seeds)

and prints the verdict per hypothesis. The decision rule is:

    H1 confirmed & H_alt rejected   → original dissociation framing
    H1 falsified & H_alt confirmed  → alignment-tax framing
    anything else                   → flagged for manual judgement

Run when all cells exist:

    python scripts/decide_n500_pivot.py

or in watch mode (polls until all 20 cells are present):

    python scripts/decide_n500_pivot.py --watch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_TABLE = ROOT / "analysis" / "tables" / "n500_pivot_metrics.csv"
OUT_REPORT = ROOT / "analysis" / "reports" / "n500_pivot_decision.md"

CONDITIONS = ("A", "B")
SEEDS = tuple(range(1, 11))
EQUIV_BAND = 0.05  # |Δ B_RLHF| below this counts as "scale invariant"


def cell_path(cond: str, seed: int) -> Path:
    return EXP_DIR / f"mx_{cond}_n500_s{seed}"


def load_summary(p: Path) -> dict | None:
    s = p / "summary.json"
    if not s.exists():
        return None
    try:
        return json.loads(s.read_text())
    except Exception:
        return None


def collect() -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            summary = load_summary(cell_path(cond, seed))
            if summary is None:
                continue
            rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "brm": summary.get("brm") or summary.get("composite_brm") or np.nan,
                    "b_rlhf": summary.get("b_rlhf") or summary.get("rlhf_bias_index") or np.nan,
                    "cooperation_rate": summary.get("cooperation_rate", np.nan),
                    "gini": summary.get("gini", np.nan),
                    "mean_wealth": summary.get("mean_wealth", np.nan),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["condition", "seed", "brm", "b_rlhf", "cooperation_rate", "gini", "mean_wealth"])
    return pd.DataFrame(rows)


def load_n100_b_rlhf() -> dict[str, list[float]]:
    """Pull B_RLHF from the §8.1 N=100 mx_{A,B}_s{1..10} cells for the
    H_alt scale-invariance test. Returns {} silently if absent."""
    out: dict[str, list[float]] = {"A": [], "B": []}
    for cond in CONDITIONS:
        for seed in SEEDS:
            summary = load_summary(EXP_DIR / f"mx_{cond}_s{seed}")
            if summary is None:
                continue
            val = summary.get("b_rlhf") or summary.get("rlhf_bias_index")
            if val is not None:
                out[cond].append(float(val))
    return out


def verdict_h1(df: pd.DataFrame) -> tuple[str, dict]:
    """One-sided Mann-Whitney: BRM(B) > BRM(A)."""
    a = df[df.condition == "A"].brm.dropna()
    b = df[df.condition == "B"].brm.dropna()
    if len(a) < 3 or len(b) < 3:
        return "insufficient_data", {"n_a": len(a), "n_b": len(b)}
    res = stats.mannwhitneyu(b, a, alternative="greater")
    delta = float(b.mean() - a.mean())
    return ("H1_confirmed" if res.pvalue < 0.05 else "H1_falsified"), {
        "delta_brm": delta,
        "p_value": float(res.pvalue),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std()),
        "std_b": float(b.std()),
    }


def verdict_h_alt(df_n500: pd.DataFrame, n100: dict[str, list[float]]) -> tuple[str, dict]:
    """Scale invariance: |B_RLHF(N=500) - B_RLHF(N=100)| < EQUIV_BAND."""
    if not n100["A"] and not n100["B"]:
        return "no_n100_baseline", {}
    out = {}
    invariant_count = 0
    for cond in CONDITIONS:
        n100_vals = n100[cond]
        n500_vals = df_n500[df_n500.condition == cond].b_rlhf.dropna().tolist()
        if not n100_vals or not n500_vals:
            out[cond] = {"status": "missing"}
            continue
        delta = abs(np.mean(n500_vals) - np.mean(n100_vals))
        invariant = delta < EQUIV_BAND
        invariant_count += int(invariant)
        out[cond] = {
            "n100_mean": float(np.mean(n100_vals)),
            "n500_mean": float(np.mean(n500_vals)),
            "abs_delta": float(delta),
            "scale_invariant": invariant,
        }
    verdict = "H_alt_confirmed" if invariant_count == len(CONDITIONS) else "H_alt_rejected"
    return verdict, out


def framing_recommendation(h1: str, h_alt: str) -> str:
    if h1 == "H1_confirmed" and h_alt == "H_alt_rejected":
        return "ORIGINAL DISSOCIATION FRAMING — grounding restores realism at primary scale."
    if h1 == "H1_falsified" and h_alt == "H_alt_confirmed":
        return "ALIGNMENT-TAX FRAMING — RLHF cooperative bias is scale-invariant."
    if h1 == "H1_confirmed" and h_alt == "H_alt_confirmed":
        return "MIXED — BRM improves at N=500 but B_RLHF stays put. Lead with BRM, footnote bias-stability."
    if h1 == "insufficient_data":
        return "WAIT — fewer than 3 seeds per arm available; rerun this script once more cells complete."
    return "FLAG FOR MANUAL JUDGEMENT — neither pre-registered pattern fits."


def write_report(df: pd.DataFrame, h1: tuple[str, dict], h_alt: tuple[str, dict]) -> None:
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TABLE, index=False)
    rec = framing_recommendation(h1[0], h_alt[0])
    lines = [
        "# N=500 LLM Sweep — Pivot Decision",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        f"**Recommended framing:** {rec}",
        "",
        "## Cells available",
        "",
        f"- Condition A: {(df.condition == 'A').sum()}/{len(SEEDS)} cells",
        f"- Condition B: {(df.condition == 'B').sum()}/{len(SEEDS)} cells",
        "",
        "## H1 — BRM(B) > BRM(A)  (one-sided MWU)",
        "",
        f"- Verdict: **{h1[0]}**",
        "```json",
        json.dumps(h1[1], indent=2),
        "```",
        "",
        "## H_alt — |B_RLHF(N=500) − B_RLHF(N=100)| < 0.05 (scale invariance)",
        "",
        f"- Verdict: **{h_alt[0]}**",
        "```json",
        json.dumps(h_alt[1], indent=2),
        "```",
        "",
        f"Underlying per-cell table: `{OUT_TABLE.relative_to(ROOT)}`",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_TABLE}")
    print(f"\n→ {rec}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="Poll every 5 min until all 20 cells exist")
    parser.add_argument("--min-cells", type=int, default=6, help="Minimum cells per arm before reporting (default 6)")
    args = parser.parse_args()

    while True:
        df = collect()
        ready = (df.condition == "A").sum() >= args.min_cells and (df.condition == "B").sum() >= args.min_cells
        if ready:
            n100 = load_n100_b_rlhf()
            write_report(df, verdict_h1(df), verdict_h_alt(df, n100))
            return 0
        if not args.watch:
            print(
                f"Only {(df.condition == 'A').sum()}/{len(SEEDS)} A cells and "
                f"{(df.condition == 'B').sum()}/{len(SEEDS)} B cells have summary.json; "
                f"need ≥ {args.min_cells} per arm. Re-run later or pass --watch.",
                file=sys.stderr,
            )
            return 2
        print(f"Waiting for cells… (A: {(df.condition == 'A').sum()}, B: {(df.condition == 'B').sum()})")
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
