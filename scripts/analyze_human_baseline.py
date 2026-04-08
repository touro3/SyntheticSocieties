#!/usr/bin/env python3
"""Analyze human-subject baseline data and build a comparison table.

Expected input CSV schema (one row per participant-round):
  participant_id, round_id, action, wealth_after
where action in {work, save, cooperate}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from metrics.behavioral_realism import compute_rlhf_bias_index
from metrics.inequality import gini_coefficient
from metrics.statistical_inference import bootstrap_ci, report_metric

VALID_ACTIONS = {"work", "save", "cooperate"}


def _write_template_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("participant_id,round_id,action,wealth_after\n", encoding="utf-8")


def _load_human_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}. "
            "Create it with columns: participant_id,round_id,action,wealth_after."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(
            f"Input CSV exists but has no rows: {path}. "
            "Add participant-round records and rerun."
        )
    required = {"participant_id", "round_id", "action", "wealth_after"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Expected participant_id, round_id, action, wealth_after."
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

    dup_rows = int(
        df.duplicated(subset=["participant_id", "round_id"], keep=False).sum()
    )

    participant_ids = [str(x) for x in df["participant_id"].dropna().unique().tolist()]
    sequential_id_pattern = _is_sequential_id_pattern(participant_ids)

    # Conservative synthetic/demo heuristic:
    # very small sample + short sessions + synthetic-looking IDs.
    synthetic_pattern_detected = bool(
        n_participants <= 10
        and max_rounds_obs <= 5
        and sequential_id_pattern
    )

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
        rows.append(
            {
                "participant_id": participant_id,
                "n_rounds": int(n),
                "coop_rate": float(dist["cooperate"]),
                "b_rlhf": float(compute_rlhf_bias_index(dist)),
                "final_wealth": float(part_sorted["wealth_after"].iloc[-1]),
            }
        )
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
    parser = argparse.ArgumentParser(
        description="Analyze human baseline data and generate comparison artifacts."
    )
    parser.add_argument("--input-csv", required=True, help="Human baseline CSV path.")
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
    args = parser.parse_args()

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
        raise ValueError(
            "Duplicate participant_id + round_id rows detected. "
            "Resolve duplicates before analysis."
        )

    if checks["synthetic_pattern_detected"] and not args.allow_synthetic:
        raise ValueError(
            "Input appears to be synthetic/demo data (small sample, short rounds, "
            "sequential placeholder IDs). Refusing publication analysis. "
            "Use --allow-synthetic only for local smoke tests."
        )

    if (
        (not checks["passes_min_participants"] or not checks["passes_min_rounds_per_participant"])
        and not args.allow_noncompliant
    ):
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
        a: (action_counts.get(a, 0) / total_actions if total_actions > 0 else 0.0)
        for a in sorted(VALID_ACTIONS)
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

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics: {output_json}")

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
