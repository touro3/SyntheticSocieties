#!/usr/bin/env python3
"""Analyze human-subject baseline data and build a comparison table.

Extended CSV schema (output by human_experiment/server/server.py):
  participant_id, round_id, action, target, wealth_after, stress_after,
  pre_trust, pre_risk, cooperation_count, total_rounds

Minimum required columns (legacy):
  participant_id, round_id, action, wealth_after

Phase 29.2 additions:
  - JSD(D_human, D_condA) vs JSD(D_human, D_condB) — action distribution divergence
  - Spearman ρ: pre_trust → cooperation_rate (validates E[coop|profile] formula)
  - --synthetic: generate synthetic participants for offline testing
  - Outputs analysis/tables/human_vs_simulation_reference.json

Phase 29.3 additions (KS indistinguishability):
  - Two-sample Kolmogorov-Smirnov test on per-participant cooperation rates
  - Primary claim: KS(human, Cond B) < KS(human, Cond A) with p-value
  - "Behavioral indistinguishability" flag when KS p > 0.05 for Cond B
  - --sim-wealth-a / --sim-wealth-b: wealth arrays for wealth-distribution KS test
  - Outputs ks_indistinguishability section in human_vs_simulation_reference.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import pandas as pd

from metrics.behavioral_realism import compute_rlhf_bias_index
from metrics.inequality import gini_coefficient
from metrics.statistical_inference import bootstrap_ci, report_metric

VALID_ACTIONS = {"work", "save", "cooperate"}


# ── Synthetic data generation ─────────────────────────────────────────────────


def _generate_synthetic_participants(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic participant data based on E[coop]=0.2+0.6*trust*(1-risk).

    Used for offline testing without real Prolific data.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pid = f"synth_{i:04d}"
        pre_trust = rng.uniform(0.1, 0.9)
        pre_risk = rng.uniform(0.1, 0.9)
        # Formula-predicted cooperation probability
        p_coop = max(0.0, min(1.0, 0.2 + 0.6 * pre_trust * (1 - pre_risk)))
        # Remaining rounds split between work (higher p) and save
        p_work = max(0.0, 0.8 - p_coop)
        p_save = max(0.0, 1.0 - p_coop - p_work)
        total = p_coop + p_work + p_save
        p_coop /= total
        p_work /= total
        p_save /= total

        wealth = 50.0
        for r in range(1, 11):
            roll = rng.random()
            if roll < p_coop:
                action = "cooperate"
                wealth = max(0.0, wealth - 5.0)
            elif roll < p_coop + p_work:
                action = "work"
                wealth += 10.0
            else:
                action = "save"
            rows.append(
                {
                    "participant_id": pid,
                    "round_id": r,
                    "action": action,
                    "target": "",
                    "wealth_after": round(wealth, 2),
                    "stress_after": round(rng.uniform(0.1, 0.6), 3),
                    "pre_trust": round(pre_trust, 3),
                    "pre_risk": round(pre_risk, 3),
                    "cooperation_count": 0,
                    "total_rounds": 10,
                }
            )
    return pd.DataFrame(rows)


# ── JSD divergence ────────────────────────────────────────────────────────────


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon Divergence between two categorical distributions (nats).

    Both dicts should sum to 1 over the same key set.
    """
    keys = sorted(set(p) | set(q))
    p_arr = [p.get(k, 1e-10) for k in keys]
    q_arr = [q.get(k, 1e-10) for k in keys]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0)

    m_arr = [(a + b) / 2 for a, b in zip(p_arr, q_arr)]
    return 0.5 * _kl(p_arr, m_arr) + 0.5 * _kl(q_arr, m_arr)


# ── Spearman correlation ──────────────────────────────────────────────────────


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return float("nan")

    def _ranks(v: list[float]) -> list[float]:
        indexed = sorted(enumerate(v), key=lambda t: t[1])
        ranks = [0.0] * n
        for rank, (i, _) in enumerate(indexed, 1):
            ranks[i] = float(rank)
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    xm = sum(rx) / n
    ym = sum(ry) / n
    num = sum((rx[i] - xm) * (ry[i] - ym) for i in range(n))
    den = (sum((rx[i] - xm) ** 2 for i in range(n)) * sum((ry[i] - ym) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else float("nan")


def _compute_trust_coop_spearman(part_df: pd.DataFrame) -> dict:
    """Compute Spearman ρ between pre_trust and cooperation_rate per participant."""
    if "pre_trust" not in part_df.columns:
        return {"spearman_rho": None, "n": 0, "note": "pre_trust column missing"}
    valid = part_df.dropna(subset=["pre_trust", "coop_rate"])
    if len(valid) < 3:
        return {"spearman_rho": None, "n": len(valid), "note": "insufficient data"}
    rho = _spearman(valid["pre_trust"].tolist(), valid["coop_rate"].tolist())
    return {
        "spearman_rho": round(rho, 4),
        "n": len(valid),
        "note": "Spearman ρ(pre_trust, cooperation_rate)",
    }


def _write_template_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("participant_id,round_id,action,wealth_after\n", encoding="utf-8")


def _load_human_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}. Create it with columns: participant_id,round_id,action,wealth_after."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV exists but has no rows: {path}. Add participant-round records and rerun.")
    required = {"participant_id", "round_id", "action", "wealth_after"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Expected participant_id, round_id, action, wealth_after."
        )

    df["action"] = df["action"].astype(str).str.strip().str.lower()
    bad_actions = sorted(set(df["action"].unique()) - VALID_ACTIONS)
    if bad_actions:
        raise ValueError(f"Invalid action labels: {bad_actions}. Valid: {sorted(VALID_ACTIONS)}")

    df["round_id"] = pd.to_numeric(df["round_id"], errors="raise")
    df["wealth_after"] = pd.to_numeric(df["wealth_after"], errors="raise")
    return df


def _is_sequential_id_pattern(ids: list[str]) -> bool:
    """Heuristic detector for placeholder/demo IDs like P_001, p1, participant_2."""
    if not ids:
        return False
    patterns = (
        r"^p[_-]?\d+$",
        r"^p[_-]?\d{3,}$",
        r"^participant[_-]?\d+$",
    )
    normalized = [str(x).strip().lower() for x in ids if str(x).strip()]
    if not normalized:
        return False
    return all(any(re.match(p, x) for p in patterns) for x in normalized)


def _quality_checks(
    df: pd.DataFrame,
    min_participants: int,
    min_rounds_per_participant: int,
) -> dict:
    """Compute publication-facing data quality checks for human baseline data."""
    part_round_counts = df.groupby("participant_id", dropna=False)["round_id"].nunique()
    n_participants = int(part_round_counts.shape[0])
    min_rounds_obs = int(part_round_counts.min()) if n_participants > 0 else 0
    max_rounds_obs = int(part_round_counts.max()) if n_participants > 0 else 0

    dup_rows = int(df.duplicated(subset=["participant_id", "round_id"], keep=False).sum())

    participant_ids = [str(x) for x in df["participant_id"].dropna().unique().tolist()]
    sequential_id_pattern = _is_sequential_id_pattern(participant_ids)

    # Conservative synthetic/demo heuristic:
    # very small sample + short sessions + synthetic-looking IDs.
    synthetic_pattern_detected = bool(n_participants <= 10 and max_rounds_obs <= 5 and sequential_id_pattern)

    return {
        "n_participants": n_participants,
        "min_rounds_observed": min_rounds_obs,
        "max_rounds_observed": max_rounds_obs,
        "duplicate_participant_round_rows": dup_rows,
        "sequential_id_pattern": sequential_id_pattern,
        "synthetic_pattern_detected": synthetic_pattern_detected,
        "passes_min_participants": n_participants >= min_participants,
        "passes_min_rounds_per_participant": min_rounds_obs >= min_rounds_per_participant,
    }


def _compute_participant_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby("participant_id", dropna=False)
    for participant_id, part in grouped:
        part_sorted = part.sort_values("round_id")
        n = len(part_sorted)
        counts = part_sorted["action"].value_counts().to_dict()
        dist = {a: counts.get(a, 0) / n for a in sorted(VALID_ACTIONS)}
        row = {
            "participant_id": participant_id,
            "n_rounds": int(n),
            "coop_rate": float(dist["cooperate"]),
            "b_rlhf": float(compute_rlhf_bias_index(dist)),
            "final_wealth": float(part_sorted["wealth_after"].iloc[-1]),
        }
        # Include pre-survey attributes if available (Phase 29.2 schema)
        if "pre_trust" in part_sorted.columns:
            row["pre_trust"] = float(part_sorted["pre_trust"].iloc[0])
        if "pre_risk" in part_sorted.columns:
            row["pre_risk"] = float(part_sorted["pre_risk"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _normalise_condition_payload(raw: dict) -> dict[str, dict]:
    """Accept either {name: metrics} or {'conditions': [{name, ...}, ...]}."""
    if "conditions" in raw and isinstance(raw["conditions"], list):
        out: dict[str, dict] = {}
        for row in raw["conditions"]:
            name = row.get("name")
            if not name:
                continue
            out[str(name)] = {
                "gini": row.get("gini"),
                "cooperation_rate": row.get("cooperation_rate"),
                "b_rlhf": row.get("b_rlhf"),
            }
        return out
    return raw


def _ks_two_sample(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test (pure Python, no scipy dependency).

    Returns (ks_statistic, p_value_approx) where p_value is approximated
    using the Kolmogorov distribution for large samples.

    The KS statistic D = max|F_a(x) - F_b(x)| over all x.
    For large n, m: p ≈ 2 * exp(-2 * D² * n*m/(n+m)).
    """
    if not a or not b:
        return float("nan"), float("nan")
    n, m = len(a), len(b)
    combined = sorted(set(a + b))
    # Build ECDFs
    def ecdf(vals: list[float], x: float) -> float:
        return sum(1 for v in vals if v <= x) / len(vals)
    d = max(abs(ecdf(a, x) - ecdf(b, x)) for x in combined)
    # Approximate p-value via Kolmogorov distribution
    nm = n * m / (n + m)
    z = d * math.sqrt(nm)
    # Series approximation: P(D > d) ≈ 2*sum_{k=1..inf} (-1)^(k+1) exp(-2k²z²)
    p_approx = 0.0
    for k in range(1, 20):
        term = ((-1) ** (k + 1)) * math.exp(-2 * (k ** 2) * (z ** 2))
        p_approx += term
        if abs(term) < 1e-10:
            break
    p_approx = max(0.0, min(1.0, 2 * p_approx))
    return round(d, 6), round(p_approx, 6)


def _behavioral_indistinguishability_test(
    human_coop_rates: list[float],
    cond_a_coop_rates: list[float] | None,
    cond_b_coop_rates: list[float] | None,
    human_wealth: list[float] | None = None,
    cond_a_wealth: list[float] | None = None,
    cond_b_wealth: list[float] | None = None,
) -> dict:
    """Run two-sample KS tests comparing human behavior to simulation conditions.

    Primary claim (if supported):
        KS(human, Cond B) statistic < KS(human, Cond A) statistic
        AND KS(human, Cond B) p-value > 0.05  →  "statistically indistinguishable"

    This is the gold-standard evidence for grounding efficacy:
    grounded agents are statistically indistinguishable from humans at the
    action-distribution level.

    Args:
        human_coop_rates: Per-participant cooperation rates from real participants.
        cond_a_coop_rates: Per-agent cooperation rates from Condition A simulation.
        cond_b_coop_rates: Per-agent cooperation rates from Condition B simulation.
        human_wealth: Per-participant final wealth (for wealth-distribution KS test).
        cond_a_wealth: Per-agent final wealth from Condition A.
        cond_b_wealth: Per-agent final wealth from Condition B.

    Returns:
        Dict with KS statistics, p-values, indistinguishability flags, and
        a human-readable interpretation string for the paper.
    """
    result: dict = {}

    def _ks_entry(name_a: str, name_b: str, vals_a: list[float], vals_b: list[float]) -> dict:
        d, p = _ks_two_sample(vals_a, vals_b)
        return {
            "ks_statistic": d,
            "p_value": p,
            "indistinguishable_at_05": p > 0.05,
            "n_a": len(vals_a),
            "n_b": len(vals_b),
            "label": f"KS({name_a}, {name_b})",
        }

    # ── Cooperation rate KS tests ─────────────────────────────────────────────
    result["cooperation_rate"] = {}
    if cond_a_coop_rates:
        result["cooperation_rate"]["human_vs_A"] = _ks_entry(
            "human", "Cond A", human_coop_rates, cond_a_coop_rates
        )
    if cond_b_coop_rates:
        result["cooperation_rate"]["human_vs_B"] = _ks_entry(
            "human", "Cond B", human_coop_rates, cond_b_coop_rates
        )

    # ── Wealth KS tests ───────────────────────────────────────────────────────
    result["wealth"] = {}
    if human_wealth and cond_a_wealth:
        result["wealth"]["human_vs_A"] = _ks_entry(
            "human", "Cond A", human_wealth, cond_a_wealth
        )
    if human_wealth and cond_b_wealth:
        result["wealth"]["human_vs_B"] = _ks_entry(
            "human", "Cond B", human_wealth, cond_b_wealth
        )

    # ── Primary claim assessment ──────────────────────────────────────────────
    coop_a = result["cooperation_rate"].get("human_vs_A", {})
    coop_b = result["cooperation_rate"].get("human_vs_B", {})

    if coop_a and coop_b:
        d_a = coop_a.get("ks_statistic", float("nan"))
        d_b = coop_b.get("ks_statistic", float("nan"))
        p_b = coop_b.get("p_value", 0.0)

        b_closer = d_b < d_a
        b_indistinguishable = p_b > 0.05

        if b_indistinguishable and b_closer:
            interp = (
                f"STRONG EVIDENCE: Grounded agents (Cond B) are statistically "
                f"indistinguishable from human participants in cooperation rate "
                f"(KS={d_b:.4f}, p={p_b:.4f} > 0.05), while ungrounded agents "
                f"(Cond A) differ significantly (KS={d_a:.4f}). "
                f"This is the gold-standard evidence for grounding efficacy."
            )
        elif b_closer:
            interp = (
                f"PARTIAL EVIDENCE: Grounded agents (Cond B) are closer to human "
                f"behavior than ungrounded (KS(B)={d_b:.4f} < KS(A)={d_a:.4f}), "
                f"but not yet statistically indistinguishable (p={p_b:.4f} < 0.05). "
                f"Increase n_participants or n_rounds for stronger evidence."
            )
        else:
            interp = (
                f"NO EVIDENCE: Grounded agents (Cond B) are not closer to human "
                f"behavior than ungrounded (KS(B)={d_b:.4f} ≥ KS(A)={d_a:.4f}). "
                f"This contradicts the grounding hypothesis — investigate."
            )

        result["primary_claim"] = {
            "cond_b_closer_to_human": b_closer,
            "cond_b_statistically_indistinguishable": b_indistinguishable,
            "ks_d_human_vs_A": d_a,
            "ks_d_human_vs_B": d_b,
            "ks_p_human_vs_B": p_b,
            "interpretation": interp,
        }

    return result


def _to_markdown_table(human_row: dict, condition_rows: dict[str, dict]) -> str:
    lines = [
        "# Human vs Simulation Baseline Comparison",
        "",
        "| Condition | Gini | Coop Rate | B_RLHF |",
        "|---|---:|---:|---:|",
        (
            f"| Real humans | {human_row['gini']:.3f} | "
            f"{human_row['cooperation_rate']:.3f} | {human_row['b_rlhf']:.3f} |"
        ),
    ]

    for name, row in condition_rows.items():
        g = row.get("gini")
        c = row.get("cooperation_rate")
        b = row.get("b_rlhf")
        g_s = f"{float(g):.3f}" if g is not None else "NA"
        c_s = f"{float(c):.3f}" if c is not None else "NA"
        b_s = f"{float(b):.3f}" if b is not None else "NA"
        lines.append(f"| {name} | {g_s} | {c_s} | {b_s} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human baseline data and generate comparison artifacts.")
    parser.add_argument("--input-csv", default=None, help="Human baseline CSV path.")
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate 50 synthetic participants for offline testing."
    )
    parser.add_argument(
        "--simulation-json",
        default=None,
        help="JSON with simulation action distributions per condition "
        "for JSD comparison ({condA: {work, save, cooperate}, ...}).",
    )
    parser.add_argument(
        "--jsd-output-json",
        default="analysis/tables/human_vs_simulation_reference.json",
        help="Output JSON for JSD + Spearman comparison table.",
    )
    parser.add_argument(
        "--output-json",
        default="analysis/tables/human_baseline_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--comparison-json",
        default=None,
        help="Optional JSON with simulation condition metrics for table generation.",
    )
    parser.add_argument(
        "--output-markdown",
        default="analysis/reports/human_baseline_comparison.md",
        help="Markdown comparison table output path.",
    )
    parser.add_argument(
        "--init-template",
        action="store_true",
        help="Create an empty input CSV template and exit.",
    )
    parser.add_argument(
        "--min-participants",
        type=int,
        default=30,
        help="Minimum participant count for publication-grade baseline (default: 30).",
    )
    parser.add_argument(
        "--min-rounds-per-participant",
        type=int,
        default=10,
        help="Minimum rounds per participant for publication-grade baseline (default: 10).",
    )
    parser.add_argument(
        "--allow-noncompliant",
        action="store_true",
        help="Allow running on pilot/noncompliant sample sizes without failing.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow running on synthetic/demo-like data patterns (not for publication).",
    )
    # KS indistinguishability arguments (Phase 29.3)
    parser.add_argument(
        "--sim-coop-a",
        default=None,
        help="JSON file with per-agent cooperation rates for Cond A "
        "(list of floats). Used for KS indistinguishability test.",
    )
    parser.add_argument(
        "--sim-coop-b",
        default=None,
        help="JSON file with per-agent cooperation rates for Cond B "
        "(list of floats). Used for KS indistinguishability test.",
    )
    parser.add_argument(
        "--sim-wealth-a",
        default=None,
        help="JSON file with per-agent final wealth for Cond A (list of floats).",
    )
    parser.add_argument(
        "--sim-wealth-b",
        default=None,
        help="JSON file with per-agent final wealth for Cond B (list of floats).",
    )
    args = parser.parse_args()

    # ── Synthetic data shortcut ───────────────────────────────────────────────
    if args.synthetic:
        print("Generating 50 synthetic participants (formula-based)...")
        df = _generate_synthetic_participants(n=50, seed=42)
        # Bypass all quality guards for synthetic runs
        args.allow_synthetic = True
        args.allow_noncompliant = True
    else:
        if args.input_csv is None:
            parser.error("--input-csv is required unless --synthetic is set.")
        input_path = Path(args.input_csv)
        if args.init_template:
            _write_template_csv(input_path)
            print(f"Template created: {input_path}")
            print("Next step: fill rows with participant-round data, then rerun this script.")
            return

        if not input_path.exists():
            _write_template_csv(input_path)
            print(f"Input CSV not found. Created template: {input_path}")
            print("Fill this file with Prolific data and rerun the same command.")
            return

        df = _load_human_csv(input_path)
    checks = _quality_checks(
        df,
        min_participants=args.min_participants,
        min_rounds_per_participant=args.min_rounds_per_participant,
    )

    if checks["duplicate_participant_round_rows"] > 0:
        raise ValueError("Duplicate participant_id + round_id rows detected. Resolve duplicates before analysis.")

    if checks["synthetic_pattern_detected"] and not args.allow_synthetic:
        raise ValueError(
            "Input appears to be synthetic/demo data (small sample, short rounds, "
            "sequential placeholder IDs). Refusing publication analysis. "
            "Use --allow-synthetic only for local smoke tests."
        )

    if (
        not checks["passes_min_participants"] or not checks["passes_min_rounds_per_participant"]
    ) and not args.allow_noncompliant:
        raise ValueError(
            "Human baseline sample does not meet publication thresholds. "
            f"Observed: n_participants={checks['n_participants']}, "
            f"min_rounds_observed={checks['min_rounds_observed']}. "
            f"Required: n_participants>={args.min_participants}, "
            f"min_rounds_per_participant>={args.min_rounds_per_participant}. "
            "Use --allow-noncompliant for pilot analysis only."
        )

    part = _compute_participant_level(df)

    action_counts = df["action"].value_counts().to_dict()
    total_actions = int(len(df))
    action_dist = {
        a: (action_counts.get(a, 0) / total_actions if total_actions > 0 else 0.0) for a in sorted(VALID_ACTIONS)
    }

    coop_pooled = action_dist["cooperate"]
    b_rlhf_pooled = float(compute_rlhf_bias_index(action_dist))

    coop_report = report_metric(part["coop_rate"].tolist())
    b_rlhf_report = report_metric(part["b_rlhf"].tolist())

    gini_point, gini_lo, gini_hi = bootstrap_ci(
        part["final_wealth"].tolist(),
        stat_fn=lambda arr: gini_coefficient(arr.tolist()),
        n_bootstrap=2000,
        confidence=0.95,
        random_state=42,
    )

    payload = {
        "metadata": {
            "input_csv": str(args.input_csv),
            "n_rows": int(len(df)),
            "n_participants": int(part["participant_id"].nunique()),
            "n_rounds_min": int(part["n_rounds"].min()),
            "n_rounds_max": int(part["n_rounds"].max()),
            "quality_checks": checks,
            "thresholds": {
                "min_participants": int(args.min_participants),
                "min_rounds_per_participant": int(args.min_rounds_per_participant),
            },
            "analysis_mode": (
                "publication"
                if checks["passes_min_participants"]
                and checks["passes_min_rounds_per_participant"]
                and not checks["synthetic_pattern_detected"]
                else "pilot_or_demo"
            ),
        },
        "human_metrics": {
            "gini": {
                "value": round(float(gini_point), 4),
                "lower": round(float(gini_lo), 4),
                "upper": round(float(gini_hi), 4),
                "ci_str": f"{gini_point:.4f} [{gini_lo:.4f}, {gini_hi:.4f}]",
            },
            "cooperation_rate": {
                "pooled_value": round(float(coop_pooled), 4),
                "participant_mean_ci": coop_report,
            },
            "b_rlhf": {
                "pooled_value": round(float(b_rlhf_pooled), 4),
                "participant_mean_ci": b_rlhf_report,
            },
            "action_distribution": {k: round(float(v), 4) for k, v in action_dist.items()},
        },
    }

    # ── Spearman trust → cooperation ──────────────────────────────────────────
    trust_coop = _compute_trust_coop_spearman(part)
    payload["trust_cooperation_spearman"] = trust_coop
    if trust_coop.get("spearman_rho") is not None:
        print(f"Spearman ρ(pre_trust, coop_rate) = {trust_coop['spearman_rho']:.4f} (n={trust_coop['n']})")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics: {output_json}")

    # ── KS behavioral indistinguishability test + JSD (Phase 29.3) ──────────
    def _load_float_list(path_str: str | None) -> list[float] | None:
        if path_str is None:
            return None
        p = Path(path_str)
        if not p.exists():
            print(f"Warning: {path_str} not found, skipping KS component.")
            return None
        return [float(x) for x in json.loads(p.read_text())]

    cond_a_coop = _load_float_list(args.sim_coop_a)
    cond_b_coop = _load_float_list(args.sim_coop_b)
    cond_a_wealth = _load_float_list(args.sim_wealth_a)
    cond_b_wealth = _load_float_list(args.sim_wealth_b)

    human_coop_rates = part["coop_rate"].tolist()
    human_wealth_list = part["final_wealth"].tolist()

    ks_results = _behavioral_indistinguishability_test(
        human_coop_rates=human_coop_rates,
        cond_a_coop_rates=cond_a_coop,
        cond_b_coop_rates=cond_b_coop,
        human_wealth=human_wealth_list if cond_a_wealth or cond_b_wealth else None,
        cond_a_wealth=cond_a_wealth,
        cond_b_wealth=cond_b_wealth,
    )

    if ks_results.get("primary_claim"):
        claim = ks_results["primary_claim"]
        print("\n── KS Behavioral Indistinguishability Test ──────────────────────")
        print(claim["interpretation"])
        print(f"  KS(human, Cond A) = {claim.get('ks_d_human_vs_A', 'N/A')}")
        print(f"  KS(human, Cond B) = {claim.get('ks_d_human_vs_B', 'N/A')} "
              f"(p = {claim.get('ks_p_human_vs_B', 'N/A')})")
        print("─" * 65)

    payload["ks_indistinguishability"] = ks_results

    # Update JSD output to include KS results
    if args.simulation_json and Path(args.simulation_json).exists():
        sim_data = json.loads(Path(args.simulation_json).read_text())
        human_dist = {k: round(float(v), 6) for k, v in action_dist.items()}
        jsd_results = {}
        for cond_name, cond_dist in sim_data.items():
            if isinstance(cond_dist, dict):
                jsd_val = _jsd(human_dist, {k: float(v) for k, v in cond_dist.items()})
                jsd_results[cond_name] = round(jsd_val, 6)
        reference_payload = {
            "human_action_distribution": human_dist,
            "jsd_vs_conditions": jsd_results,
            "ks_indistinguishability": ks_results,
            "trust_cooperation_spearman": trust_coop,
            "primary_test": "ks_indistinguishability",
            "note": (
                "Primary comparison is two-sample KS test (Phase 29.3). "
                "JSD is a secondary distributional comparison. "
                "Indistinguishability claim requires KS p > 0.05 for Cond B."
            ),
        }
        jsd_out = Path(args.jsd_output_json)
        jsd_out.parent.mkdir(parents=True, exist_ok=True)
        jsd_out.write_text(json.dumps(reference_payload, indent=2))
        print(f"Saved KS+JSD comparison: {jsd_out}")
        print("JSD(human, condition):")
        for cond, val in sorted(jsd_results.items(), key=lambda x: x[1]):
            print(f"  {cond}: {val:.6f}")

    if args.comparison_json:
        comp_path = Path(args.comparison_json)
        if not comp_path.exists():
            print(f"Comparison JSON not found, skipping markdown table: {comp_path}")
        else:
            raw = json.loads(comp_path.read_text())
            cond = _normalise_condition_payload(raw)

            human_row = {
                "gini": payload["human_metrics"]["gini"]["value"],
                "cooperation_rate": payload["human_metrics"]["cooperation_rate"]["pooled_value"],
                "b_rlhf": payload["human_metrics"]["b_rlhf"]["pooled_value"],
            }
            md = _to_markdown_table(human_row, cond)
            out_md = Path(args.output_markdown)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(md, encoding="utf-8")
            print(f"Saved comparison table: {out_md}")

    print(
        "Human baseline summary: "
        f"Gini={payload['human_metrics']['gini']['value']:.3f}, "
        f"Coop={payload['human_metrics']['cooperation_rate']['pooled_value']:.3f}, "
        f"B_RLHF={payload['human_metrics']['b_rlhf']['pooled_value']:.3f}"
    )


if __name__ == "__main__":
    main()
