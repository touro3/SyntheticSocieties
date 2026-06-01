"""Continuous trust-gradient correlation at seed level (n=20).

Bypasses the n=4 group-rank power ceiling documented in §6.5 / Limitation #4
by computing Spearman ρ over individual seed runs rather than group means.

Reads `analysis/tables/trust_gradient.json` (5 seeds × 4 trust bands = 20 runs)
and writes `analysis/tables/trust_gradient_continuous.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
IN_PATH = REPO / "analysis" / "tables" / "trust_gradient.json"
OUT_PATH = REPO / "analysis" / "tables" / "trust_gradient_continuous.json"


def main() -> None:
    data = json.loads(IN_PATH.read_text())
    runs = data["all_runs"]

    trust = np.array([r["ess_reference_trust"] for r in runs])
    coop = np.array([r["coop_rate"] for r in runs])

    spearman = stats.spearmanr(trust, coop)
    pearson = stats.pearsonr(trust, coop)
    kendall = stats.kendalltau(trust, coop)

    # Bootstrap 95% CI on Spearman ρ
    rng = np.random.default_rng(42)
    n = len(runs)
    boot_rhos = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        boot_rhos.append(stats.spearmanr(trust[idx], coop[idx]).statistic)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    out = {
        "n_observations": int(n),
        "design": "seed-level continuous correlation (5 seeds × 4 ESS trust bands)",
        "spearman": {
            "rho": float(spearman.statistic),
            "p_value": float(spearman.pvalue),
            "ci95_bootstrap": [float(ci_lo), float(ci_hi)],
        },
        "pearson": {
            "r": float(pearson.statistic),
            "p_value": float(pearson.pvalue),
        },
        "kendall": {
            "tau_b": float(kendall.statistic),
            "p_value": float(kendall.pvalue),
        },
        "comparison_to_group_level": {
            "group_n": data["correlation"]["n_groups"],
            "group_rho": data["correlation"]["spearman_r"],
            "group_p": data["correlation"]["p_value"],
            "min_achievable_p_at_n4": 2 / 24,
            "note": (
                "Seed-level continuous design uses n=20 observations (5 seeds × 4 bands), "
                "bypassing the structural floor of 2/4!=0.083 imposed by the group-rank design."
            ),
        },
        "audit_row": "A.5 (extended)",
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"✓ Continuous trust-gradient correlation → {OUT_PATH}")
    print(f"  n = {n}")
    print(
        f"  Spearman ρ = {out['spearman']['rho']:.3f}  "
        f"p = {out['spearman']['p_value']:.4f}  "
        f"95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]"
    )
    print(f"  Pearson  r = {out['pearson']['r']:.3f}  p = {out['pearson']['p_value']:.4f}")
    print(f"  Kendall τ  = {out['kendall']['tau_b']:.3f}  p = {out['kendall']['p_value']:.4f}")


if __name__ == "__main__":
    main()
