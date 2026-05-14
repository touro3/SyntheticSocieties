"""H9: out-of-sample behavioural cross-cultural benchmark.

Tests whether BGF simulated cooperation rates correlate with **published
public-goods-game contribution rates** rather than the ESS/WVS trust-attitude
benchmarks used in §8.3. This is the out-of-sample test §9 Limitation 11
asks for: an independent behavioural reference rather than another trust survey.

Benchmark source
----------------
Herrmann, B., Thöni, C., & Gächter, S. (2008). "Antisocial Punishment Across
Societies." Science 319(5868): 1362–1367. Supplementary materials, Table S1:
period-1 mean contributions (MU out of 20 endowment) in the standard public-
goods game across 16 worldwide subject pools.

Cluster mapping (best-effort geographic/cultural proxy):
  - nordic    → Copenhagen          12.16
  - northern  → Zurich + St. Gallen 11.45
  - anglo     → Nottingham + Boston 11.57
  - western   → Bonn                 9.79
  - southern  → Athens               5.69
  - eastern   → Minsk + Samara + Dnipro  7.84

These mappings are documented and conservative (no cherry-picking of high or
low cities within a cluster). The cluster→city map is the only researcher
degree of freedom in this analysis; the contribution numbers themselves are
fixed by Herrmann et al. 2008.

Output: analysis/tables/h9_cross_cultural_behavioral.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SIM_CSV = REPO / "analysis" / "tables" / "cross_cultural_expanded_correlation.csv"
OUT_JSON = REPO / "analysis" / "tables" / "h9_cross_cultural_behavioral.json"

# Herrmann et al. 2008 period-1 PGG mean contributions (MU out of 20).
HERRMANN_2008_CITIES = {
    "boston": 10.96,
    "bonn": 9.79,
    "copenhagen": 12.16,
    "dnipro": 7.42,
    "athens": 5.69,
    "minsk": 7.18,
    "nottingham": 12.18,
    "samara": 8.93,
    "st_gallen": 11.43,
    "zurich": 11.46,
}

CLUSTER_TO_CITIES = {
    "nordic": ["copenhagen"],
    "northern": ["zurich", "st_gallen"],
    "anglo": ["nottingham", "boston"],
    "western": ["bonn"],
    "southern": ["athens"],
    "eastern": ["minsk", "samara", "dnipro"],
}


def cluster_benchmark(cluster: str) -> float:
    cities = CLUSTER_TO_CITIES[cluster]
    return float(np.mean([HERRMANN_2008_CITIES[c] for c in cities]))


def main() -> None:
    sim = pd.read_csv(SIM_CSV)
    sim["herrmann_pgg_contrib"] = sim["cluster"].map(cluster_benchmark)

    coop = sim["mean_coop"].values
    bench = sim["herrmann_pgg_contrib"].values

    spearman = stats.spearmanr(bench, coop)
    pearson = stats.pearsonr(bench, coop)
    kendall = stats.kendalltau(bench, coop)

    # Exact two-tailed permutation p for Spearman at n=6
    from itertools import permutations

    rho_obs = spearman.statistic
    ranks_coop = stats.rankdata(coop)
    n = len(coop)
    rho_perms = []
    for perm in permutations(range(n)):
        rho_p = stats.spearmanr(bench, ranks_coop[list(perm)]).statistic
        rho_perms.append(rho_p)
    rho_perms = np.array(rho_perms)
    exact_p_two = float(np.mean(np.abs(rho_perms) >= abs(rho_obs)))

    out = {
        "hypothesis": "H9: simulated cooperation rate Spearman ρ with Herrmann 2008 PGG contributions",
        "n_clusters": int(n),
        "benchmark_source": "Herrmann, Thöni & Gächter 2008, Science (PGG period-1 contributions)",
        "cluster_mapping": CLUSTER_TO_CITIES,
        "per_cluster": [
            {
                "cluster": row.cluster,
                "simulated_coop_rate": float(row.mean_coop),
                "herrmann_pgg_contrib_MU": float(row.herrmann_pgg_contrib),
                "cities_used": CLUSTER_TO_CITIES[row.cluster],
            }
            for row in sim.itertuples(index=False)
        ],
        "spearman": {
            "rho": float(spearman.statistic),
            "asymptotic_p": float(spearman.pvalue),
            "exact_permutation_p_two_tailed": exact_p_two,
        },
        "pearson": {
            "r": float(pearson.statistic),
            "p_value": float(pearson.pvalue),
        },
        "kendall": {
            "tau_b": float(kendall.statistic),
            "p_value": float(kendall.pvalue),
        },
        "interpretation_note": (
            "This tests against an *independent behavioural* benchmark (PGG laboratory "
            "contributions) rather than a trust-attitude survey. Significance directly "
            "addresses Limitation 11 (circularity) — the BGF grounding uses ESS trust "
            "attitudes, never PGG contributions, so any correlation here is genuinely "
            "out-of-sample."
        ),
        "audit_row": "D.3 / Limitation 11 partial resolution",
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"✓ H9 cross-cultural behavioural benchmark → {OUT_JSON}")
    print(f"  n = {n} clusters")
    print(
        f"  Spearman ρ = {spearman.statistic:.3f}  "
        f"asymptotic p = {spearman.pvalue:.4f}  "
        f"exact permutation p (two-tailed) = {exact_p_two:.4f}"
    )
    print(f"  Pearson  r = {pearson.statistic:.3f}  p = {pearson.pvalue:.4f}")
    print(f"  Kendall τ  = {kendall.statistic:.3f}  p = {kendall.pvalue:.4f}")


if __name__ == "__main__":
    main()
