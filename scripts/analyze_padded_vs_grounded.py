"""Comparative analysis: Condition P (padded) vs Condition A and B.

Phase 28 / TOP_TIER_RESEARCH Section 1.

Usage:
    python scripts/analyze_padded_vs_grounded.py \
        --cond-a  experiments/cond_a_s42 experiments/cond_a_s123 experiments/cond_a_s7 \
        --cond-b  experiments/cond_b_s42 experiments/cond_b_s123 experiments/cond_b_s7 \
        --cond-p  experiments/padded_control_s42 experiments/padded_control_s123 experiments/padded_control_s7 \
        --out     analysis/tables/padded_control_comparison.json

Outputs:
  analysis/tables/padded_control_comparison.json   — full comparison table
  analysis/figures/padded_control_brm.png          — BRM bar chart
  analysis/figures/padded_control_b_rlhf.png       — B_RLHF bar chart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from metrics.behavioral_realism import compute_rlhf_bias_index
from metrics.inequality import gini_coefficient


# ── Core metric computation ──────────────────────────────────────────────────


def compute_condition_metrics(exp_dir: Path) -> dict:
    """Compute BRM/B_RLHF/Gini metrics from a single experiment directory.

    Reads ``events.jsonl`` produced by the simulation kernel.

    Args:
        exp_dir: Path to the experiment directory (must contain events.jsonl).

    Returns:
        Dict with keys: b_rlhf, gini, coop_rate, n_rounds, n_agents.
    """
    events_path = exp_dir / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"events.jsonl not found in {exp_dir}")

    actions: list[str] = []
    wealth_by_round: dict[int, list[float]] = {}

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)

            # Extract action type
            action_raw = event.get("action", {})
            if isinstance(action_raw, dict):
                at = action_raw.get("action_type", "")
            elif isinstance(action_raw, str):
                at = action_raw
            else:
                at = ""
            if at in ("work", "save", "cooperate"):
                actions.append(at)

            # Extract wealth for Gini
            state_after = event.get("state_after", {})
            if isinstance(state_after, dict):
                wealth = state_after.get("wealth")
                if wealth is not None:
                    round_id = event.get("round_id", 0)
                    wealth_by_round.setdefault(round_id, []).append(float(wealth))

    if not actions:
        return {"b_rlhf": 0.0, "gini": 0.0, "coop_rate": 0.0, "n_rounds": 0, "n_agents": 0}

    # B_RLHF from action distribution
    n = len(actions)
    action_dist = {
        "work": actions.count("work") / n,
        "save": actions.count("save") / n,
        "cooperate": actions.count("cooperate") / n,
    }
    b_rlhf = compute_rlhf_bias_index(action_dist)
    coop_rate = action_dist["cooperate"]

    # Gini from last available round's wealth snapshot
    gini = 0.0
    if wealth_by_round:
        last_round = max(wealth_by_round)
        final_wealth = wealth_by_round[last_round]
        if len(final_wealth) >= 2:
            gini = float(gini_coefficient(final_wealth))

    n_rounds = len(wealth_by_round)
    n_agents = max((len(v) for v in wealth_by_round.values()), default=0)

    return {
        "b_rlhf": round(b_rlhf, 4),
        "gini": round(gini, 4),
        "coop_rate": round(coop_rate, 4),
        "n_rounds": n_rounds,
        "n_agents": n_agents,
    }


def aggregate_metrics(exp_dirs: list[Path]) -> dict:
    """Aggregate metrics across multiple seeds.

    Returns mean ± std for each metric.
    """
    all_metrics: list[dict] = []
    for d in exp_dirs:
        try:
            m = compute_condition_metrics(d)
            all_metrics.append(m)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not all_metrics:
        return {}

    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    result: dict = {"n_seeds": len(all_metrics)}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        result[k] = round(float(np.mean(vals)), 4)
        result[f"{k}_std"] = round(float(np.std(vals)), 4)
        result[f"{k}_values"] = [round(v, 4) for v in vals]

    return result


# ── Statistical tests ────────────────────────────────────────────────────────


def mann_whitney_test(a: list[float], b: list[float]) -> dict:
    """Run Mann-Whitney U test comparing two groups.

    Args:
        a: Observations from group A.
        b: Observations from group B.

    Returns:
        Dict with u_statistic, p_value, and interpretation.
    """
    from scipy import stats

    if len(a) < 2 or len(b) < 2:
        return {"u_statistic": float("nan"), "p_value": float("nan"), "note": "insufficient data"}

    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "u_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant_at_05": p_val < 0.05,
    }


def cohen_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns 0.0 if both groups have zero variance.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0

    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    n_a, n_b = len(a), len(b)

    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    if pooled_std == 0.0:
        # Both groups have zero within-group variance.
        # If means differ, effect is infinite (report large sentinel); else 0.
        return float("inf") if mean_a != mean_b else 0.0

    return float((mean_a - mean_b) / pooled_std)


# ── Comparison table ─────────────────────────────────────────────────────────


def build_comparison_table(
    condition_a: dict,
    condition_p: dict,
    condition_b: dict,
) -> dict:
    """Build the full comparison table with per-condition metrics and P-vs-A / P-vs-B stats.

    For each metric, computes:
      - Mean ± std per condition
      - Mann-Whitney U (P vs A, P vs B)
      - Cohen's d (P vs A, P vs B)
      - Interpretation: whether content effect (B > P) or length effect (P > A) is detected
    """
    table: dict = {
        "condition_a": condition_a,
        "condition_p": condition_p,
        "condition_b": condition_b,
        "pa_comparison": {},  # P vs A: isolates length effect
        "pb_comparison": {},  # P vs B: isolates content effect
        "interpretation": {},
    }

    metrics_to_compare = ["b_rlhf", "gini", "coop_rate"]

    for metric in metrics_to_compare:
        vals_a = condition_a.get(f"{metric}_values", [condition_a.get(metric, 0.0)])
        vals_p = condition_p.get(f"{metric}_values", [condition_p.get(metric, 0.0)])
        vals_b = condition_b.get(f"{metric}_values", [condition_b.get(metric, 0.0)])

        pa = mann_whitney_test(vals_p, vals_a)
        pa["cohen_d"] = round(cohen_d(vals_p, vals_a), 4)

        pb = mann_whitney_test(vals_p, vals_b)
        pb["cohen_d"] = round(cohen_d(vals_p, vals_b), 4)

        table["pa_comparison"][metric] = pa
        table["pb_comparison"][metric] = pb

    # High-level interpretation for B_RLHF (primary metric)
    b_rlhf_a = condition_a.get("b_rlhf", 0.5)
    b_rlhf_p = condition_p.get("b_rlhf", 0.5)
    b_rlhf_b = condition_b.get("b_rlhf", 0.5)

    length_effect = abs(b_rlhf_p - b_rlhf_a) > 0.05  # P differs from A
    content_effect = abs(b_rlhf_b - b_rlhf_p) > 0.05  # B differs from P

    table["interpretation"]["length_effect_detected"] = length_effect
    table["interpretation"]["content_effect_detected"] = content_effect
    if content_effect and not length_effect:
        table["interpretation"]["conclusion"] = (
            "Content effect confirmed: ESS grounding drives behavioral realism, not prompt length."
        )
    elif content_effect and length_effect:
        table["interpretation"]["conclusion"] = (
            "Both content and length effects detected. Report decomposition."
        )
    elif not content_effect and not length_effect:
        table["interpretation"]["conclusion"] = (
            "Neither content nor length effect detected. Investigate simulation parameters."
        )
    else:
        table["interpretation"]["conclusion"] = (
            "Length effect only. Review grounding pipeline — ESS content may not be effective."
        )

    return table


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_comparison(table: dict, figures_dir: Path) -> None:
    """Generate bar charts comparing Condition A, P, B for key metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("b_rlhf", "B_RLHF (↓ better: less cooperation bias)"),
        ("coop_rate", "Cooperation Rate"),
        ("gini", "Gini Coefficient"),
    ]

    for metric_key, label in metrics:
        conditions = ["A (Ungrounded)", "P (Padded Control)", "B (Grounded)"]
        vals = [
            table["condition_a"].get(metric_key, 0.0),
            table["condition_p"].get(metric_key, 0.0),
            table["condition_b"].get(metric_key, 0.0),
        ]
        stds = [
            table["condition_a"].get(f"{metric_key}_std", 0.0),
            table["condition_p"].get(f"{metric_key}_std", 0.0),
            table["condition_b"].get(f"{metric_key}_std", 0.0),
        ]
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(conditions, vals, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor="black")
        ax.set_ylabel(label)
        ax.set_title(f"Condition Comparison: {label}")
        ax.set_ylim(0, max(vals) * 1.3 + 0.05)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out_path = figures_dir / f"padded_control_{metric_key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved figure: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Condition P (padded) vs A and B."
    )
    parser.add_argument(
        "--cond-a", nargs="+", required=True, metavar="DIR",
        help="Condition A experiment directories (one per seed).",
    )
    parser.add_argument(
        "--cond-b", nargs="+", required=True, metavar="DIR",
        help="Condition B experiment directories (one per seed).",
    )
    parser.add_argument(
        "--cond-p", nargs="+", required=True, metavar="DIR",
        help="Condition P experiment directories (one per seed).",
    )
    parser.add_argument(
        "--out", type=str,
        default="analysis/tables/padded_control_comparison.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--figures-dir", type=str,
        default="analysis/figures",
        help="Directory for output figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dirs_a = [Path(d) for d in args.cond_a]
    dirs_b = [Path(d) for d in args.cond_b]
    dirs_p = [Path(d) for d in args.cond_p]

    print(f"Aggregating Condition A ({len(dirs_a)} seeds)...")
    metrics_a = aggregate_metrics(dirs_a)
    print(f"Aggregating Condition B ({len(dirs_b)} seeds)...")
    metrics_b = aggregate_metrics(dirs_b)
    print(f"Aggregating Condition P ({len(dirs_p)} seeds)...")
    metrics_p = aggregate_metrics(dirs_p)

    table = build_comparison_table(metrics_a, metrics_p, metrics_b)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"\nComparison table saved to: {out_path}")

    print("\n── Interpretation ──────────────────────────────────────────")
    print(table["interpretation"]["conclusion"])
    print(f"  B_RLHF  A={metrics_a.get('b_rlhf', 'N/A')}  "
          f"P={metrics_p.get('b_rlhf', 'N/A')}  B={metrics_b.get('b_rlhf', 'N/A')}")
    print(f"  Gini    A={metrics_a.get('gini', 'N/A')}  "
          f"P={metrics_p.get('gini', 'N/A')}  B={metrics_b.get('gini', 'N/A')}")
    print(f"  CoopR   A={metrics_a.get('coop_rate', 'N/A')}  "
          f"P={metrics_p.get('coop_rate', 'N/A')}  B={metrics_b.get('coop_rate', 'N/A')}")

    plot_comparison(table, Path(args.figures_dir))


if __name__ == "__main__":
    main()
