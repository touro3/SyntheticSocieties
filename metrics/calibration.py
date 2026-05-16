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

    # Gini — delegate to canonical implementation
    from metrics.inequality import gini_coefficient as _gini

    if n > 1 and arr.sum() > 0:
        gini = _gini(arr)
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

    Statistical note: With only one held-out evaluation seed (seed=7),
    the calibration-evaluation gap estimate has zero degrees of freedom
    for variance estimation — it is a point estimate with no uncertainty
    bounds. The LOW/MEDIUM/HIGH risk labels below should be treated as
    directional signals, not statistically validated claims. A minimum of
    5 held-out seeds is recommended for reliable generalization assessment.
    """
    if policies is None:
        policies = ["llm", "template", "rule_based", "random"]

    lines = []
    lines.append("=" * 70)
    lines.append("  Calibration vs Evaluation Report")
    lines.append(f"  Calibration seeds: {CALIBRATION_SEEDS}")
    lines.append(f"  Evaluation seeds:  {EVALUATION_SEEDS}")
    lines.append("  WARNING: n=1 evaluation seed — gap estimates have no")
    lines.append("  variance bounds. Risk labels are directional only.")
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
            f"  {label:<18} {'CAL':<8} {cal['mean_wealth']:>10.1f} {cal['gini']:>8.3f} {cal['coop_rate'] * 100:>7.1f}%"
        )
        lines.append(
            f"  {'':<18} {'EVAL':<8} {ev['mean_wealth']:>10.1f} {ev['gini']:>8.3f} {ev['coop_rate'] * 100:>7.1f}%"
        )

        avg_gap = np.mean(list(gap.values()))
        risk = "LOW" if avg_gap < 10 else "MEDIUM" if avg_gap < 25 else "HIGH"
        lines.append(f"  {'':<18} {'GAP':<8} {'':>10} {'':>8} {'':>8} {avg_gap:>7.1f}%  ← {risk} risk")
        lines.append("")

    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)
    return report


# ── Calibration JSD: simulated wealth vs ESS income reference ────────────────
#
# Phase 28.1. The legacy calibration_evaluation_split() above is a 3-seed
# cal/eval design with explicitly zero uncertainty bounds. `calibration_jsd`
# instead gives a *per-run* scalar that the ten-seed confirmatory pipeline
# aggregates across seeds with a BCa 95% CI, so calibration becomes a headline
# metric on equal footing with cooperation_rate / wealth_gini / b_rlhf.

ESS_DISTRIBUTIONS = Path("data/empirical_distributions.json")


def ess_wealth_reference(
    path: str | Path = ESS_DISTRIBUTIONS,
    n: int = 1000,
) -> np.ndarray:
    """Deterministic ESS income reference (household net income decile).

    The raw per-respondent ESS parquet is not always importable (no pandas in
    minimal envs), so the reference is reconstructed from the published
    quantile summary of ``demographics.income_decile`` via piecewise-linear
    inverse-CDF interpolation. No RNG — identical on every call.

    Returns a length-``n`` float array on the 1–10 decile scale.
    """
    p_anchor = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    # ESS-9 fallback if the file/keys are missing (matches the committed
    # data/empirical_distributions.json income_decile summary).
    v_anchor = [1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0]
    try:
        with Path(path).open() as f:
            inc = json.load(f)["demographics"]["income_decile"]
        q = inc["quantiles"]
        v_anchor = [
            float(inc["min"]),
            float(q["0.1"]),
            float(q["0.25"]),
            float(q["0.5"]),
            float(q["0.75"]),
            float(q["0.9"]),
            float(inc["max"]),
        ]
    except (OSError, KeyError, ValueError, TypeError):
        pass  # fall back to the committed-summary anchors above
    # Enforce monotone non-decreasing support for a valid inverse CDF.
    v_anchor = np.maximum.accumulate(v_anchor)
    return np.interp(np.linspace(0.0, 1.0, n), p_anchor, v_anchor).astype(float)


def _minmax01(values) -> np.ndarray:
    """Min-max scale to [0, 1], dropping NaNs. A constant input maps to all
    0.5 (a single occupied bin) so JSD stays well-defined."""
    a = np.asarray(list(values), dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        raise ValueError("Expected at least one non-NaN value.")
    span = float(a.max() - a.min())
    if span <= 0.0:
        return np.full(a.shape, 0.5)
    return (a - a.min()) / span


def calibration_jsd(sim_wealth, ess_ref=None) -> float:
    """Jensen–Shannon divergence between the simulated end-state wealth
    distribution and the ESS income reference.

    Both arrays are min-max normalized to [0, 1] *independently* before
    binning, so the metric is a scale-invariant distribution-*shape*
    divergence: simulated wealth (~10–10⁴) and ESS income deciles (1–10)
    would otherwise live in disjoint histogram bins and the raw JSD would
    saturate near its maximum regardless of fit. Lower = better calibrated.
    """
    from metrics.distribution import jensen_shannon_divergence

    if ess_ref is None:
        ess_ref = ess_wealth_reference()
    sim_n = _minmax01(sim_wealth)
    ess_n = _minmax01(ess_ref)
    return float(jensen_shannon_divergence(sim_n, ess_n))


if __name__ == "__main__":
    calibration_report()
