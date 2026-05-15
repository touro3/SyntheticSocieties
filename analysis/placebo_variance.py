"""3-way variance decomposition for the placebo / semantic-isolation ablation.

Reuses the proven one-way ANOVA in ``analysis.variance_decomposition.decompose``
(it is already k-condition generic) over the three arms produced by
``scripts/run_placebo_ablation.py``:

    grounded  vs  placebo        → the SEMANTIC component
                                    (does sociological coherence matter?)
    placebo   vs  unconditioned  → the ANY-CONDITIONING component
                                    (does prompt heterogeneity alone matter?)

Reading the contrast
--------------------
If realism (cooperation_rate etc.) drops sharply grounded→placebo but barely
moves placebo→unconditioned, the gain is genuinely semantic and the central
BGF claim is defended. If grounded≈placebo, the apparent gain was just prompt
entropy and the claim is unsafe.

CPU-only. Reads ``experiments/placebo_abl_*/summary.json``.
Output: ``analysis/tables/placebo_variance.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.variance_decomposition import decompose  # noqa: E402

EXPERIMENTS = REPO / "experiments"
OUT_JSON = REPO / "analysis" / "tables" / "placebo_variance.json"

ARM_ORDER = ["grounded", "placebo", "unconditioned"]
METRICS = ["cooperation_rate", "wealth_gini", "work_rate", "save_rate"]


def _extract_metrics(summary: dict) -> dict[str, float]:
    """Pull the comparison metrics out of a run's summary.json.

    Falls back gracefully across the ``event_behavior`` / ``behavior`` /
    ``wealth`` blocks written by ``metrics.summary``.
    """
    eb = summary.get("event_behavior", {}) or {}
    beh = summary.get("behavior", {}) or {}
    wealth = summary.get("wealth", {}) or {}
    return {
        "cooperation_rate": eb.get("cooperation_rate", beh.get("cooperation_rate", float("nan"))),
        "wealth_gini": wealth.get("gini", float("nan")),
        "work_rate": eb.get("work_rate", float("nan")),
        "save_rate": eb.get("save_rate", float("nan")),
    }


def collect(prefix: str = "placebo_abl_") -> pd.DataFrame:
    """Build a tidy frame: one row per run, columns = arm + metrics."""
    rows = []
    for d in sorted(EXPERIMENTS.glob(f"{prefix}*")):
        sfile = d / "summary.json"
        if not sfile.exists():
            continue
        # exp id pattern: placebo_abl_<arm>_s<seed>
        tail = d.name[len(prefix) :]
        arm = tail.rsplit("_s", 1)[0]
        if arm not in ARM_ORDER:
            continue
        summary = json.loads(sfile.read_text())
        rows.append({"experiment_id": d.name, "condition": arm, **_extract_metrics(summary)})
    return pd.DataFrame(rows)


def main() -> None:
    df = collect()
    if df.empty:
        print("No placebo_abl_* experiments found. Run scripts/run_placebo_ablation.py first.")
        sys.exit(1)

    present = [a for a in ARM_ORDER if a in set(df["condition"])]
    print(f"Arms present: {present}  |  runs: {len(df)}")
    print(f"{'metric':<20}{'eta^2':>8}{'F':>9}{'p':>9}  condition means")
    print("-" * 78)

    results = {}
    for metric in METRICS:
        sub = df.dropna(subset=[metric])
        if sub["condition"].nunique() < 2:
            continue
        r = decompose(sub, metric, condition_col="condition")
        results[metric] = r
        means = " ".join(f"{a}={r['condition_means'].get(a, float('nan')):.3f}" for a in present)
        print(f"{metric:<20}{r['eta_squared']:>8.3f}{r['F']:>9.2f}{r['p_anova']:>9.4f}  {means}")

    out = {
        "model": "one-way ANOVA over 3 arms (grounded / placebo / unconditioned)",
        "interpretation": (
            "Compare condition_means: a large grounded→placebo drop with a "
            "small placebo→unconditioned drop isolates the SEMANTIC effect of "
            "ESS grounding from mere prompt heterogeneity. High eta^2 means the "
            "arm contrast dominates seed replication noise."
        ),
        "n_runs": int(len(df)),
        "arms_present": present,
        "decomposition": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n✓ {OUT_JSON}")


if __name__ == "__main__":
    main()
