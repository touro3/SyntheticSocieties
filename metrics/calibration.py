"""
Calibration vs Evaluation separation for BGF experiments.

Splits experiment data into calibration and evaluation sets
by seed to detect overfitting or instability:

  - Calibration: seeds 42, 123 (used for tuning / analysis)
  - Evaluation:  seed 7 (held out for validation)

Usage:
    python -c "from metrics.calibration import calibration_report; calibration_report()"
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


CALIBRATION_SEEDS = [42, 123]
EVALUATION_SEEDS = [7]
EXPERIMENTS_DIR = Path("experiments")

POLICY_PREFIX = {
    "llm": "llm",
    "template": "template",
    "rule_based": "rule",
    "random": "random",
}


def load_summary(exp_id: str) -> dict:
    path = EXPERIMENTS_DIR / exp_id / "summary.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.loads(f.read())


def compute_metrics(exp_ids: list[str]) -> dict:
    """Compute aggregate metrics across a set of experiments."""
    all_wealth = []
    all_actions = Counter()

    for exp_id in exp_ids:
        s = load_summary(exp_id)
        if not s:
            continue
        wealth_vals = s.get("wealth", {}).get("values", [])
        all_wealth.extend(wealth_vals)
        eac = s.get("event_action_counts", {})
        all_actions.update(eac)

    if not all_wealth:
        return {"mean_wealth": 0.0, "gini": 0.0, "coop_rate": 0.0, "n_agents": 0}

    arr = np.array(all_wealth)
    n = len(arr)

    # Gini
    if n > 1 and arr.sum() > 0:
        diff_sum = sum(abs(arr[i] - arr[j]) for i in range(n) for j in range(n))
        gini = float(diff_sum / (2 * n * n * arr.mean()))
    else:
        gini = 0.0

    total_a = max(sum(all_actions.values()), 1)
    coop_rate = all_actions.get("cooperate", 0) / total_a

    return {
        "mean_wealth": float(np.mean(arr)),
        "std_wealth": float(np.std(arr)),
        "gini": gini,
        "coop_rate": coop_rate,
        "action_counts": dict(all_actions),
        "n_agents": n,
    }


def calibration_evaluation_split(policy: str) -> dict:
    """
    Split experiments for a policy into calibration and evaluation sets.
    Returns metrics for each set and the gap.
    """
    prefix = POLICY_PREFIX.get(policy, policy)

    cal_ids = [f"cmp_{prefix}_s{s}" for s in CALIBRATION_SEEDS]
    eval_ids = [f"cmp_{prefix}_s{s}" for s in EVALUATION_SEEDS]

    cal_metrics = compute_metrics(cal_ids)
    eval_metrics = compute_metrics(eval_ids)

    # Compute calibration-evaluation gap
    gap = {}
    for key in ["mean_wealth", "gini", "coop_rate"]:
        cal_val = cal_metrics.get(key, 0.0)
        eval_val = eval_metrics.get(key, 0.0)
        if cal_val > 0:
            gap[f"{key}_gap_pct"] = abs(eval_val - cal_val) / cal_val * 100
        else:
            gap[f"{key}_gap_pct"] = 0.0

    return {
        "policy": policy,
        "calibration": cal_metrics,
        "evaluation": eval_metrics,
        "gap": gap,
    }


def calibration_report(policies: list[str] = None) -> str:
    """
    Generate calibration vs evaluation report for all policies.
    Returns formatted string.
    """
    if policies is None:
        policies = ["llm", "template", "rule_based", "random"]

    lines = []
    lines.append("=" * 70)
    lines.append("  Calibration vs Evaluation Report")
    lines.append(f"  Calibration seeds: {CALIBRATION_SEEDS}")
    lines.append(f"  Evaluation seeds:  {EVALUATION_SEEDS}")
    lines.append("=" * 70)
    lines.append("")

    header = f"{'Policy':<20} {'Set':<8} {'Wealth μ':>10} {'Gini':>8} {'Coop%':>8} {'Gap%':>8}"
    lines.append(header)
    lines.append("─" * 70)

    for policy in policies:
        result = calibration_evaluation_split(policy)
        cal = result["calibration"]
        ev = result["evaluation"]
        gap = result["gap"]

        label = {
            "llm": "LLM (Mistral)",
            "template": "Template",
            "rule_based": "Rule-Based",
            "random": "Random",
        }.get(policy, policy)

        if cal["n_agents"] == 0 and ev["n_agents"] == 0:
            lines.append(f"  {label:<18} {'N/A':<8} {'(no data)':>10}")
            continue

        lines.append(
            f"  {label:<18} {'CAL':<8} "
            f"{cal['mean_wealth']:>10.1f} {cal['gini']:>8.3f} "
            f"{cal['coop_rate'] * 100:>7.1f}%"
        )
        lines.append(
            f"  {'':<18} {'EVAL':<8} "
            f"{ev['mean_wealth']:>10.1f} {ev['gini']:>8.3f} "
            f"{ev['coop_rate'] * 100:>7.1f}%"
        )

        avg_gap = np.mean(list(gap.values()))
        risk = "LOW" if avg_gap < 10 else "MEDIUM" if avg_gap < 25 else "HIGH"
        lines.append(
            f"  {'':<18} {'GAP':<8} "
            f"{'':>10} {'':>8} {'':>8} "
            f"{avg_gap:>7.1f}%  ← {risk} risk"
        )
        lines.append("")

    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    calibration_report()
