#!/usr/bin/env python3
"""Research-integrity audit for BGF experiment artifacts.

Purpose:
- Catch stale/mixed analytics
- Validate basic plausibility ranges
- Verify holdout and human-baseline readiness for publication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _is_valid_range(series: pd.Series, lo: float, hi: float) -> bool:
    if series.empty:
        return False
    s = pd.to_numeric(series, errors="coerce")
    return bool(((s >= lo) & (s <= hi)).all())


def _append(out: dict, severity: str, msg: str) -> None:
    out[severity].append(msg)


def _audit_manifest(path: Path, out: dict) -> dict:
    payload = _load_json(path)
    args = payload.get("args", {})
    if not isinstance(args, dict):
        _append(out, "blockers", "Manifest is missing 'args' object.")
        return {}

    seeds = args.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        _append(out, "blockers", "Manifest has no seed list.")
    if args.get("analytics_scope") != "run":
        _append(out, "warnings", "Analytics scope is not 'run' (historical mixing risk).")

    exp_ids = payload.get("experiment_ids", [])
    if not exp_ids:
        _append(out, "warnings", "Manifest has no experiment_ids list.")

    return args


def _infer_policy_from_exp_id(exp_id: str) -> str:
    raw = exp_id.lower()
    if "template" in raw:
        return "template"
    if "rule" in raw:
        return "rule_based"
    if "random" in raw:
        return "random"
    if "llm" in raw:
        return "llm"
    return "unknown"


def _audit_action_distributions(manifest_path: Path, out: dict) -> None:
    """Detect severe single-action collapse from per-run summary.json files."""
    payload = _load_json(manifest_path)
    exp_ids = payload.get("experiment_ids", []) or []
    collapse_hits: dict[str, int] = {}
    totals: dict[str, int] = {}

    for exp_id in exp_ids:
        summary_path = Path("experiments") / str(exp_id) / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue
        counts = summary.get("event_action_counts", {})
        total = int(sum(int(v) for v in counts.values())) if counts else 0
        if total <= 0:
            continue
        dominant = max(int(v) for v in counts.values()) / total
        policy = _infer_policy_from_exp_id(str(exp_id))
        totals[policy] = totals.get(policy, 0) + 1
        if dominant >= 0.95:
            collapse_hits[policy] = collapse_hits.get(policy, 0) + 1

    for policy, n_total in totals.items():
        n_collapse = collapse_hits.get(policy, 0)
        frac = n_collapse / n_total if n_total > 0 else 0.0
        if frac >= 0.40:
            _append(
                out,
                "warnings",
                (
                    f"{policy}: {n_collapse}/{n_total} runs show severe action collapse "
                    "(>=95% one action)."
                ),
            )


def _audit_policy_tables(
    llm_vs_baselines_path: Path,
    policy_comparison_path: Path,
    expected_seed_count: int,
    include_llm: bool,
    out: dict,
) -> None:
    df = _load_csv(llm_vs_baselines_path)
    if df.empty:
        _append(out, "blockers", "llm_vs_baselines.csv is empty.")
        return

    req_policies = ["template", "rule_based", "random"] + (["llm"] if include_llm else [])
    present = set(df["policy_type"].astype(str).tolist())
    missing = [p for p in req_policies if p not in present]
    if missing:
        _append(out, "blockers", f"Missing policy rows in llm_vs_baselines.csv: {missing}")

    for policy in req_policies:
        rows = df[df["policy_type"] == policy]
        if rows.empty:
            continue
        n_runs = int(rows["n_runs"].iloc[0])
        if n_runs != expected_seed_count:
            _append(
                out,
                "blockers",
                f"Policy '{policy}' has n_runs={n_runs}, expected {expected_seed_count}.",
            )

    if "avg_gini" in df.columns and not _is_valid_range(df["avg_gini"], 0.0, 1.0):
        _append(out, "blockers", "avg_gini contains values outside [0, 1].")

    if "avg_wealth_mean" in df.columns:
        w = pd.to_numeric(df["avg_wealth_mean"], errors="coerce")
        if (w <= 0).any():
            _append(out, "warnings", "Some avg_wealth_mean values are non-positive.")

    if policy_comparison_path.exists():
        pcmp = _load_csv(policy_comparison_path)
        if "avg_stress_mean" in pcmp.columns and not _is_valid_range(
            pcmp["avg_stress_mean"], -1.0, 1.0
        ):
            _append(out, "warnings", "avg_stress_mean has values outside [-1, 1].")


def _audit_cross_cultural(path: Path, level: str, out: dict) -> None:
    if not path.exists():
        msg = "cross_cultural_expanded_results.json not found."
        if level == "publication":
            _append(out, "blockers", msg)
        else:
            _append(out, "warnings", msg)
        return

    payload = _load_json(path)
    fit = payload.get("benchmark_fit", {})
    if "wvs" not in fit:
        _append(out, "blockers", "Cross-cultural output missing WVS holdout fit.")
        return
    wvs = fit.get("wvs", {})
    pearson = float(wvs.get("pearson_r", 0.0))
    spearman = float(wvs.get("spearman_rho", 0.0))
    if pearson <= 0 or spearman <= 0:
        _append(
            out,
            "warnings",
            f"WVS holdout fit is weak/non-positive (pearson={pearson:.3f}, spearman={spearman:.3f}).",
        )

    has_control = bool(payload.get("metadata", {}).get("contains_ungrounded_control", False))
    has_cmp = "condition_comparison" in payload
    if level == "publication" and (not has_control or not has_cmp):
        _append(
            out,
            "blockers",
            "Publication mode requires grounded-vs-ungrounded holdout comparison.",
        )


def _audit_human_baseline(path: Path, level: str, out: dict) -> None:
    if not path.exists():
        msg = "human_baseline_metrics.json not found."
        if level == "publication":
            _append(out, "blockers", msg)
        else:
            _append(out, "warnings", msg)
        return

    payload = _load_json(path)
    meta = payload.get("metadata", {})
    checks = meta.get("quality_checks", {})
    mode = str(meta.get("analysis_mode", "unknown"))
    if not checks:
        msg = "Human baseline metrics are legacy/incomplete (missing quality_checks metadata)."
        if level == "publication":
            _append(out, "blockers", msg)
        else:
            _append(out, "warnings", msg)
    else:
        if checks.get("synthetic_pattern_detected"):
            if level == "publication":
                _append(out, "blockers", "Human baseline is flagged as synthetic/demo-like.")
            else:
                _append(out, "warnings", "Human baseline is flagged as synthetic/demo-like.")
        if checks.get("duplicate_participant_round_rows", 0) > 0:
            _append(out, "blockers", "Human baseline has duplicate participant-round rows.")
    if level == "publication":
        if mode != "publication":
            _append(
                out,
                "blockers",
                f"Human baseline analysis mode is '{mode}', expected 'publication'.",
            )


def _write_markdown(report: dict, out_path: Path) -> None:
    lines = [
        "# Research Integrity Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Level: `{report['level']}`",
        "",
        "## Blockers",
    ]
    if report["blockers"]:
        lines.extend(f"- {x}" for x in report["blockers"])
    else:
        lines.append("- None")

    lines += ["", "## Warnings"]
    if report["warnings"]:
        lines.extend(f"- {x}" for x in report["warnings"])
    else:
        lines.append("- None")

    lines += ["", "## Info"]
    if report["info"]:
        lines.extend(f"- {x}" for x in report["info"])
    else:
        lines.append("- None")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit experiment artifacts for trustworthiness.")
    parser.add_argument(
        "--manifest",
        default="analysis/reports/last_pipeline_run_manifest.json",
        help="Path to run manifest JSON.",
    )
    parser.add_argument(
        "--llm-vs-baselines",
        default="analysis/tables/llm_vs_baselines.csv",
        help="Path to llm_vs_baselines.csv.",
    )
    parser.add_argument(
        "--policy-comparison",
        default="analysis/tables/policy_comparison.csv",
        help="Path to policy_comparison.csv.",
    )
    parser.add_argument(
        "--cross-cultural",
        default="analysis/cross_cultural_expanded_results.json",
        help="Path to cross-cultural expanded JSON.",
    )
    parser.add_argument(
        "--human-baseline",
        default="analysis/tables/human_baseline_metrics.json",
        help="Path to human baseline metrics JSON.",
    )
    parser.add_argument(
        "--level",
        choices=["basic", "publication"],
        default="basic",
        help="Audit strictness level.",
    )
    parser.add_argument(
        "--output-json",
        default="analysis/reports/research_integrity_audit.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-markdown",
        default="analysis/reports/research_integrity_audit.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit non-zero if blockers are found.",
    )
    args = parser.parse_args()

    report = {
        "level": args.level,
        "status": "PASS",
        "blockers": [],
        "warnings": [],
        "info": [],
    }

    manifest_args = _audit_manifest(Path(args.manifest), report)
    expected_seed_count = len(manifest_args.get("seeds", [])) if manifest_args else 0
    include_llm = bool(manifest_args.get("include_llm", True))
    if expected_seed_count <= 0:
        expected_seed_count = 1
        _append(
            report,
            "warnings",
            "Could not infer expected seed count from manifest; using fallback=1.",
        )

    _audit_policy_tables(
        Path(args.llm_vs_baselines),
        Path(args.policy_comparison),
        expected_seed_count=expected_seed_count,
        include_llm=include_llm,
        out=report,
    )
    _audit_action_distributions(Path(args.manifest), report)
    _audit_cross_cultural(Path(args.cross_cultural), args.level, report)
    _audit_human_baseline(Path(args.human_baseline), args.level, report)

    if report["blockers"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "PASS_WITH_WARNINGS"

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, Path(args.output_markdown))

    print(f"Audit status: {report['status']}")
    print(f"Blockers: {len(report['blockers'])} | Warnings: {len(report['warnings'])}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {args.output_markdown}")

    if args.fail_on_blockers and report["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
