#!/usr/bin/env python3
"""Analyse a cross-model cultural-evolution sweep: stats + trajectory figure.

Reads an analysis/evolution/<tag>_results.json produced by run_evolution.py and
emits:
  * analysis/evolution/<tag>_stats.json   (per-model mean/sd, MWU, collapse rate)
  * analysis/figures/<tag>.png            (cooperation vs generation, per model)

Usage:
    python scripts/analyze_evolution.py analysis/evolution/crossmodel_claude_gpt4o_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIG_DIR = PROJECT_ROOT / "analysis" / "figures"

_COLLAPSE_THRESHOLD = 0.5  # a lineage "collapsed" if final cooperation < this


def _per_model(runs: list[dict]) -> dict[str, dict]:
    models = sorted({r["model"] for r in runs})
    out = {}
    for m in models:
        mr = [r for r in runs if r["model"] == m]
        finals = np.array([r["final_cooperation"] for r in mr])
        trajs = np.array([r["cooperation_trajectory"] for r in mr])
        collapsed = int((finals < _COLLAPSE_THRESHOLD).sum())
        out[m] = {
            "n_seeds": len(mr),
            "final_mean": round(float(finals.mean()), 4),
            "final_sd": round(float(finals.std()), 4),
            "collapse_rate": round(collapsed / len(mr), 4),
            "n_collapsed": collapsed,
            "mean_trajectory": trajs.mean(axis=0).round(4).tolist(),
            "finals": finals.round(4).tolist(),
        }
    return out


def _compare(runs: list[dict], a: str, b: str) -> dict:
    """Two-model comparison: MWU on finals + Fisher exact on collapse rate."""
    from scipy.stats import fisher_exact, mannwhitneyu

    fa = [r["final_cooperation"] for r in runs if r["model"] == a]
    fb = [r["final_cooperation"] for r in runs if r["model"] == b]
    res: dict = {"models": [a, b]}
    try:
        U, p = mannwhitneyu(fa, fb, alternative="greater")
        res["mwu_U"] = round(float(U), 2)
        res["mwu_p_a_gt_b"] = round(float(p), 4)
    except ValueError as exc:
        res["mwu_error"] = str(exc)
    # Collapse contingency: [[a_ok, a_collapsed],[b_ok, b_collapsed]]
    a_c = sum(1 for x in fa if x < _COLLAPSE_THRESHOLD)
    b_c = sum(1 for x in fb if x < _COLLAPSE_THRESHOLD)
    table = [[len(fa) - a_c, a_c], [len(fb) - b_c, b_c]]
    odds, fp = fisher_exact(table, alternative="greater")
    res["collapse_contingency"] = table
    res["fisher_p"] = round(float(fp), 4)
    return res


def _figure(per_model: dict, tag: str, runs: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"claude": "#C15F3C", "gpt4o": "#10A37F", "gemini": "#4285F4", "mistral": "#7A5CFF"}
    for m, stats in per_model.items():
        col = colors.get(m, None)
        # faint individual seed trajectories
        for r in runs:
            if r["model"] == m:
                ax.plot(r["cooperation_trajectory"], color=col, alpha=0.12, linewidth=0.8)
        ax.plot(
            stats["mean_trajectory"],
            color=col,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=f"{m} (final {stats['final_mean']:.2f}±{stats['final_sd']:.2f}, "
            f"collapse {stats['n_collapsed']}/{stats['n_seeds']})",
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation (donation) rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Cultural evolution of cooperation in the Donor Game")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = _FIG_DIR / f"{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="Path to <tag>_results.json")
    args = ap.parse_args()

    results_path = Path(args.results)
    data = json.loads(results_path.read_text())
    runs = data["runs"]
    tag = results_path.stem.replace("_results", "")

    per_model = _per_model(runs)
    models = list(per_model)
    stats = {
        "tag": tag,
        "metadata": data.get("metadata", {}),
        "per_model": per_model,
    }
    if len(models) == 2:
        # Order so the higher-cooperation model is "a" for the one-sided MWU.
        a, b = sorted(models, key=lambda m: per_model[m]["final_mean"], reverse=True)
        stats["comparison"] = _compare(runs, a, b)

    stats_path = results_path.parent / f"{tag}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    fig_path = _figure(per_model, tag, runs)

    print(f"Stats:  {stats_path}")
    print(f"Figure: {fig_path}")
    for m, s in per_model.items():
        print(f"  {m:10s} final={s['final_mean']:.3f}±{s['final_sd']:.3f}  collapse={s['n_collapsed']}/{s['n_seeds']}")
    if "comparison" in stats:
        c = stats["comparison"]
        print(
            f"  {c['models'][0]} > {c['models'][1]}: MWU p={c.get('mwu_p_a_gt_b')}, Fisher(collapse) p={c['fisher_p']}"
        )


if __name__ == "__main__":
    sys.exit(main())
