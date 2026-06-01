#!/usr/bin/env python
"""§8.1.4 multi-seed confidence-bands figure.

Reads ``experiments/mx_A_s{1..10}/summary.json`` and ``mx_B_s{1..10}``
(or any explicit list of cells passed via --pattern), computes per-metric
bootstrap 95% CIs across seeds, and writes
``analysis/figures/multi_seed_bands.png``.

Default is the §8.1 N=100 / T=30 / 10-seed Mistral comparison; pass
``--pattern 'mx_{cond}_n500_s{seed}' --seeds 1..10`` once the N=500 cells
land.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_FIG = ROOT / "analysis" / "figures" / "multi_seed_bands.png"

METRIC_PATHS = {
    "brm": ("metrics", "brm"),
    "cooperation_rate": ("metrics", "cooperation_rate"),
    "gini": ("metrics", "gini"),
    "mean_wealth": ("metrics", "mean_wealth"),
}
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def get(summary: dict, path: tuple[str, ...]) -> float | None:
    cur: object = summary
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return float(cur) if isinstance(cur, (int, float)) else None


def bootstrap_ci(vals: list[float], q_lo: float = 0.025, q_hi: float = 0.975) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 2:
        m = float(arr.mean()) if len(arr) else float("nan")
        return m, m, m
    resamples = rng.choice(arr, size=(N_BOOTSTRAP, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(resamples, q_lo)), float(np.quantile(resamples, q_hi))


def collect(pattern: str, seeds: list[int]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {"A": {m: [] for m in METRIC_PATHS}, "B": {m: [] for m in METRIC_PATHS}}
    for cond in ("A", "B"):
        for seed in seeds:
            path = EXP_DIR / pattern.format(cond=cond, seed=seed) / "summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text())
            for metric, addr in METRIC_PATHS.items():
                v = get(summary, addr)
                if v is not None:
                    out[cond][metric].append(v)
    return out


def make_plot(data: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    metrics = list(METRIC_PATHS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    colors = {"A": "#d62728", "B": "#1f77b4"}
    for ax, metric in zip(axes, metrics):
        for i, cond in enumerate(("A", "B")):
            vals = data[cond][metric]
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            ax.errorbar(
                i,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=colors[cond],
                capsize=6,
                markersize=8,
                label=f"Condition {cond} (n={len(vals)})",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A", "B"])
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("metric value (mean ± 95% bootstrap CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("§8.1 multi-seed A vs B (bootstrap 95% CI, n_boot=2000)", y=1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


def parse_seeds(spec: str) -> list[int]:
    if ".." in spec:
        lo, hi = spec.split("..", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="mx_{cond}_s{seed}", help="Cell name template (default: §8.1 N=100)")
    p.add_argument("--seeds", default="1..10")
    p.add_argument("--out", type=Path, default=OUT_FIG)
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    data = collect(args.pattern, seeds)
    counts = {c: {m: len(v) for m, v in data[c].items()} for c in ("A", "B")}
    print("Cells found per arm/metric:", counts)
    if all(len(data[c][m]) == 0 for c in ("A", "B") for m in METRIC_PATHS):
        print("No cells found for pattern. Aborting.", file=sys.stderr)
        return 2
    make_plot(data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
