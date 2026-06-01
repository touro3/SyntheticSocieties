"""Analyze vignette realism ratings collected via /human-eval/rating endpoint.

Reads data/human/prolific_ratings.jsonl, computes per-vignette and aggregate
statistics, and generates analysis/figures/human_eval_boxplot.png.

Usage:
    python scripts/analyze_human_eval_ratings.py
    python scripts/analyze_human_eval_ratings.py --input data/human/prolific_ratings.jsonl
    python scripts/analyze_human_eval_ratings.py --min-participants 5   # dev mode
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_DEFAULT_INPUT = Path("data/human/prolific_ratings.jsonl")
_DEFAULT_OUTPUT_JSON = Path("analysis/tables/human_eval_vignette_metrics.json")
_DEFAULT_OUTPUT_PNG = Path("analysis/figures/human_eval_boxplot.png")
_MIN_PARTICIPANTS = 30


def load_ratings(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found: {path}")
    records = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  Warning: skipping malformed line {lineno}: {exc}", file=sys.stderr)
    return records


def compute_metrics(records: list[dict]) -> dict:
    by_vignette: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_vignette[r["vignette_id"]].append(r)

    vignette_stats = {}
    all_a, all_b = [], []

    for vid, recs in sorted(by_vignette.items()):
        a_scores = [r["realism_a"] for r in recs]
        b_scores = [r["realism_b"] for r in recs]
        all_a.extend(a_scores)
        all_b.extend(b_scores)

        n = len(recs)
        pref_counts = {"A": 0, "B": 0, "tie": 0}
        for r in recs:
            pref_counts[r.get("preferred", "tie")] += 1

        vignette_stats[vid] = {
            "n": n,
            "mean_a": round(float(np.mean(a_scores)), 3),
            "mean_b": round(float(np.mean(b_scores)), 3),
            "std_a": round(float(np.std(a_scores, ddof=1)) if n > 1 else 0.0, 3),
            "std_b": round(float(np.std(b_scores, ddof=1)) if n > 1 else 0.0, 3),
            "prefer_a_pct": round(pref_counts["A"] / n * 100, 1),
            "prefer_b_pct": round(pref_counts["B"] / n * 100, 1),
            "prefer_tie_pct": round(pref_counts["tie"] / n * 100, 1),
        }

    n_participants = len({r.get("prolific_pid", "") for r in records})
    overall = {
        "n_ratings": len(records),
        "n_participants": n_participants,
        "mean_realism_a": round(float(np.mean(all_a)), 3) if all_a else None,
        "mean_realism_b": round(float(np.mean(all_b)), 3) if all_b else None,
        "b_preferred_pct": round(sum(1 for r in records if r.get("preferred") == "B") / len(records) * 100, 1)
        if records
        else None,
    }

    return {"overall": overall, "vignettes": vignette_stats}


def plot_boxplot(records: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation", file=sys.stderr)
        return

    by_vignette: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    vignette_ids = sorted({r["vignette_id"] for r in records})

    for r in records:
        by_vignette[r["vignette_id"]][0].append(r["realism_a"])
        by_vignette[r["vignette_id"]][1].append(r["realism_b"])

    fig, axes = plt.subplots(1, len(vignette_ids), figsize=(4 * len(vignette_ids), 5), sharey=True)
    if len(vignette_ids) == 1:
        axes = [axes]

    colors = {"A": "#5b8dee", "B": "#4caf50"}

    for ax, vid in zip(axes, vignette_ids):
        a_scores, b_scores = by_vignette[vid]
        bp = ax.boxplot(
            [a_scores, b_scores],
            tick_labels=["Agent A\n(ungrounded)", "Agent B\n(ESS-grounded)"],
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"color": "#888"},
            capprops={"color": "#888"},
            flierprops={"marker": "o", "markerfacecolor": "#888", "markersize": 4},
        )
        bp["boxes"][0].set_facecolor(colors["A"])
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(colors["B"])
        bp["boxes"][1].set_alpha(0.75)

        ax.set_title(f"Vignette {vid}", fontsize=11)
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks(range(1, 8))
        ax.tick_params(labelsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Realism rating (1–7)", fontsize=10)

        mean_a = np.mean(a_scores) if a_scores else 0
        mean_b = np.mean(b_scores) if b_scores else 0
        ax.scatter([1, 2], [mean_a, mean_b], color="white", s=40, zorder=5)

    fig.suptitle(
        "Human Evaluation: Agent Realism Ratings (Condition A vs B)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BGF human evaluation vignette ratings")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=_DEFAULT_OUTPUT_PNG)
    parser.add_argument(
        "--min-participants",
        type=int,
        default=_MIN_PARTICIPANTS,
        help="Minimum participants required for publication analysis",
    )
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help="Continue even if min_participants threshold not met (dev mode)",
    )
    args = parser.parse_args()

    print(f"Loading ratings: {args.input}")
    records = load_ratings(args.input)
    print(f"  {len(records)} records loaded")

    metrics = compute_metrics(records)
    n_part = metrics["overall"]["n_participants"]

    if n_part < args.min_participants:
        msg = (
            f"Only {n_part} participants — need ≥{args.min_participants} for publication claims. "
            f"Pass --allow-insufficient to proceed anyway."
        )
        if not args.allow_insufficient:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics: {args.output_json}")

    ov = metrics["overall"]
    print(
        f"\nOverall: n_participants={n_part}, "
        f"mean_A={ov['mean_realism_a']}, mean_B={ov['mean_realism_b']}, "
        f"B_preferred={ov['b_preferred_pct']}%"
    )
    for vid, vs in metrics["vignettes"].items():
        print(
            f"  {vid}: A={vs['mean_a']}±{vs['std_a']}  B={vs['mean_b']}±{vs['std_b']}  prefer_B={vs['prefer_b_pct']}%"
        )

    plot_boxplot(records, args.output_png)


if __name__ == "__main__":
    main()
