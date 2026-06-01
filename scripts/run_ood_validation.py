#!/usr/bin/env python3
"""Out-of-distribution (OOD) realism validation — leave-one-cluster-out.

Breaks the BRM-circularity objection: the held-out cluster's published ESS-11
trust benchmark NEVER enters the trust→behavior calibration. We fit the
calibration on the train clusters only, then ask whether the held-out cluster's
*simulated* cooperation falls where the train-fitted line predicts from its
(unseen) benchmark trust. Small OOD error ⇒ the framework generalizes rather
than merely echoing its conditioning source.

Data scoping (see population.ood_split): the local parquet is AT-only, so the
split necessarily operates at the cluster-benchmark level — the only level
with cross-country variation available.

Reuse: cluster simulation is delegated to ``scripts.run_cross_cultural._run_cluster``
(unchanged) and the cluster set to ``population.country_clusters.load_clusters``.

Usage
-----
    # Dry-run (synthetic cluster outcomes, no GPU, no parquet needed):
    python scripts/run_ood_validation.py --dry-run --held-out eastern

    # Mock policy on real ESS cohorts (CPU):
    python scripts/run_ood_validation.py --held-out nordic --policy mock

    # Full LLM run (GPU):
    python scripts/run_ood_validation.py --held-out eastern --policy llm --seeds 42,123,7

Output: analysis/tables/ood_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from population.country_clusters import load_clusters  # noqa: E402
from population.ood_split import leave_one_cluster_out  # noqa: E402
from scripts.run_cross_cultural import _run_cluster  # noqa: E402  (reuse, unchanged)

OUT_JSON = PROJECT_ROOT / "analysis" / "tables" / "ood_validation.json"


def _mean_coop(cluster, seeds, n_agents, n_rounds, policy, dry_run) -> float:
    """Mean simulated cooperation rate for a cluster across seeds."""
    rates = [
        _run_cluster(cluster, n_agents, n_rounds, policy, seed, dry_run).simulated_cooperation_rate for seed in seeds
    ]
    return float(np.mean(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCO out-of-distribution realism validation.")
    parser.add_argument("--held-out", default="eastern", help="Cluster held out for OOD eval.")
    parser.add_argument("--policy", default="mock", choices=["mock", "rule_based", "llm"])
    parser.add_argument("--dry-run", action="store_true", help="Synthetic outcomes; no parquet/GPU.")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seeds", default="42,123,7")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    split = leave_one_cluster_out(args.held_out)
    all_by_name = {c.name: c for c in load_clusters()}

    print(f"OOD leave-one-cluster-out: eval='{args.held_out}'")
    print(f"  train clusters: {[c.name for c in split.train_clusters]}")
    print(f"  eval  cluster : {[c.name for c in split.eval_clusters]}")
    print(f"  policy={args.policy} dry_run={args.dry_run} seeds={seeds}\n")

    # ── Train: simulate train clusters, fit coop ~ a + b * ess_mean_trust ──
    train_pts = []
    for c in split.train_clusters:
        coop = _mean_coop(c, seeds, args.agents, args.rounds, args.policy, args.dry_run)
        train_pts.append((c.name, c.ess_mean_trust, coop))
    tx = np.array([p[1] for p in train_pts])
    ty = np.array([p[2] for p in train_pts])

    if np.allclose(tx, tx[0]):
        print("Degenerate: train trust benchmarks have no variation; cannot fit.")
        sys.exit(1)
    b, a = np.polyfit(tx, ty, 1)  # slope, intercept

    # ── Eval: simulate the held-out cluster, score against UNSEEN benchmark ──
    eval_results = []
    for c in split.eval_clusters:
        sim_coop = _mean_coop(c, seeds, args.agents, args.rounds, args.policy, args.dry_run)
        pred_coop = float(a + b * c.ess_mean_trust)  # benchmark never seen in fit
        abs_err = abs(sim_coop - pred_coop)
        # Bounded realism score in [0,1]: 1 = perfect OOD generalization.
        ood_realism = max(0.0, 1.0 - abs_err / max(1e-9, np.std(ty) or 1.0))
        eval_results.append(
            {
                "cluster": c.name,
                "ess_mean_trust_unseen": c.ess_mean_trust,
                "simulated_cooperation": round(sim_coop, 4),
                "predicted_from_train_fit": round(pred_coop, 4),
                "ood_abs_error": round(abs_err, 4),
                "ood_realism_score": round(ood_realism, 4),
            }
        )

    out = {
        "design": "leave-one-cluster-out (LOCO); eval benchmark excluded from calibration fit",
        "data_scope": "cluster-benchmark level (local parquet is AT-only) — see population.ood_split",
        "held_out": args.held_out,
        "policy": args.policy,
        "dry_run": args.dry_run,
        "seeds": seeds,
        "train_fit": {"slope": round(float(b), 5), "intercept": round(float(a), 5)},
        "train_points": [{"cluster": n, "ess_mean_trust": t, "sim_coop": round(co, 4)} for n, t, co in train_pts],
        "eval": eval_results,
        "interpretation": (
            "ood_abs_error is the gap between the held-out cluster's simulated "
            "cooperation and the value predicted by a calibration fitted WITHOUT "
            "its benchmark. Small error ⇒ realism generalizes out-of-distribution, "
            "refuting the BRM-circularity objection."
        ),
        "all_clusters_known": sorted(all_by_name),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    for r in eval_results:
        print(
            f"  {r['cluster']:<9} sim={r['simulated_cooperation']:.3f} "
            f"pred={r['predicted_from_train_fit']:.3f} "
            f"|err|={r['ood_abs_error']:.3f} realism={r['ood_realism_score']:.3f}"
        )
    print(f"\n✓ {OUT_JSON}")


if __name__ == "__main__":
    main()
