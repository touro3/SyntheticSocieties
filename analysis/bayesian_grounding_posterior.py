"""Beta-Binomial conjugate posterior on grounding effect.

Treat each round-agent decision as a Bernoulli draw of "cooperated or not".
Under a Beta(1,1) (uniform) prior, the posterior over cooperation rate is
Beta(α + k, β + n − k) where k is the number of cooperate actions and n the
total action count.

We compute the posterior for Condition A and Condition B separately, then
compute the posterior probability that the B cooperation rate is *closer
to the empirical PGG range [0.35, 0.65]* than A — the metric that matters
for the grounding hypothesis.

CPU-only. Output: analysis/tables/bayesian_grounding_posterior.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SEED_CSV = REPO / "analysis" / "tables" / "grounding_comparison_seed_metrics.csv"
OUT_JSON = REPO / "analysis" / "tables" / "bayesian_grounding_posterior.json"


def posterior_from_rate(coop_rate: float, n_rounds: int, n_agents: int) -> dict:
    """Beta(1+k, 1+n-k) posterior from observed rate."""
    n = max(1, int(n_rounds * n_agents))
    k = int(round(coop_rate * n))
    a, b = 1 + k, 1 + n - k
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    sd = float(np.sqrt(var))
    ci_lo = float(stats.beta.ppf(0.025, a, b))
    ci_hi = float(stats.beta.ppf(0.975, a, b))
    return {
        "n_trials": n,
        "k_successes": k,
        "alpha_post": a,
        "beta_post": b,
        "post_mean": mean,
        "post_sd": sd,
        "hdi95_lo": ci_lo,
        "hdi95_hi": ci_hi,
    }


def main() -> None:
    df = pd.read_csv(SEED_CSV)
    df = df[df["condition_key"].isin(["pure_llm_ess_persona", "grounded_llm_ess_persona"])].copy()

    # Reconstruct trial counts from rate × N agents × T rounds. The seed CSV
    # does not include explicit round counts so we approximate from rate
    # precision — for s42 the original run was N=50 T=10 (paper §6.1); s43/s44
    # are N=20 T=5 short-horizon. Use union counts.
    seed_runs = {
        42: {"pure_llm_ess_persona": (50, 10), "grounded_llm_ess_persona": (20, 8)},
        43: {"pure_llm_ess_persona": (20, 5), "grounded_llm_ess_persona": (20, 5)},
        44: {"pure_llm_ess_persona": (20, 5), "grounded_llm_ess_persona": (20, 5)},
    }

    posteriors: dict[str, list[dict]] = {"A_ungrounded": [], "B_grounded": []}
    for _, row in df.iterrows():
        seed = int(row.seed)
        cond_key = row.condition_key
        bucket = "A_ungrounded" if cond_key == "pure_llm_ess_persona" else "B_grounded"
        n_ag, n_rd = seed_runs.get(seed, {}).get(cond_key, (20, 5))
        post = posterior_from_rate(float(row.cooperation_rate), n_rd, n_ag)
        post["seed"] = seed
        post["coop_rate_observed"] = float(row.cooperation_rate)
        posteriors[bucket].append(post)

    # Pooled posterior per condition (sum α-1 and β-1 across seeds; combine
    # with the uniform prior once).
    def pool(rows: list[dict]) -> dict:
        k = sum(r["k_successes"] for r in rows)
        n = sum(r["n_trials"] for r in rows)
        a, b = 1 + k, 1 + n - k
        return posterior_from_rate(k / n, n, 1) | {
            "alpha_post": a,
            "beta_post": b,
            "post_mean": a / (a + b),
            "hdi95_lo": float(stats.beta.ppf(0.025, a, b)),
            "hdi95_hi": float(stats.beta.ppf(0.975, a, b)),
        }

    pooled_a = pool(posteriors["A_ungrounded"])
    pooled_b = pool(posteriors["B_grounded"])

    # Posterior probability that B is closer to the empirical PGG band [0.35, 0.65]
    # than A — Monte-Carlo from posteriors.
    rng = np.random.default_rng(42)
    M = 200_000
    samples_a = rng.beta(pooled_a["alpha_post"], pooled_a["beta_post"], size=M)
    samples_b = rng.beta(pooled_b["alpha_post"], pooled_b["beta_post"], size=M)
    band = (0.35, 0.65)

    def closeness(x: np.ndarray) -> np.ndarray:
        # distance from band: 0 inside, |x − nearest_edge| outside
        below = np.maximum(0, band[0] - x)
        above = np.maximum(0, x - band[1])
        return below + above

    closer_b = float(np.mean(closeness(samples_b) < closeness(samples_a)))
    p_b_in_band = float(np.mean((samples_b >= band[0]) & (samples_b <= band[1])))
    p_a_in_band = float(np.mean((samples_a >= band[0]) & (samples_a <= band[1])))
    p_b_higher_than_a = float(np.mean(samples_b > samples_a))

    out = {
        "prior": "Beta(1,1) uniform",
        "likelihood": "Binomial(n_trials, p_coop)",
        "per_seed_posteriors": posteriors,
        "pooled_posteriors": {"A_ungrounded": pooled_a, "B_grounded": pooled_b},
        "decision_quantities": {
            "P(B_coop > A_coop)": p_b_higher_than_a,
            "P(B_coop in empirical [0.35,0.65])": p_b_in_band,
            "P(A_coop in empirical [0.35,0.65])": p_a_in_band,
            "P(B closer to empirical band than A)": closer_b,
        },
        "mc_samples": M,
        "audit_row": "B.bayes (new)",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("Pooled posteriors:")
    print(f"  A: mean={pooled_a['post_mean']:.4f}  95% HDI=[{pooled_a['hdi95_lo']:.4f}, {pooled_a['hdi95_hi']:.4f}]")
    print(f"  B: mean={pooled_b['post_mean']:.4f}  95% HDI=[{pooled_b['hdi95_lo']:.4f}, {pooled_b['hdi95_hi']:.4f}]")
    print("Decision quantities:")
    print(f"  P(B > A)                     = {p_b_higher_than_a:.4f}")
    print(f"  P(B in [0.35, 0.65])         = {p_b_in_band:.4f}")
    print(f"  P(A in [0.35, 0.65])         = {p_a_in_band:.4f}")
    print(f"  P(B closer to band than A)   = {closer_b:.4f}")
    print(f"\n✓ {OUT_JSON}")


if __name__ == "__main__":
    main()
