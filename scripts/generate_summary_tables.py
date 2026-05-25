"""Generate statistical summary tables from experiment results.

Reads experiment summary.json files across all runs and produces
CSV tables suitable for inclusion in the capstone paper.

Usage:
    python scripts/generate_summary_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
TABLES_DIR = ROOT / "analysis" / "tables"

sys.path.insert(0, str(ROOT))


def load_experiment_summaries() -> list[dict]:
    """Load summary.json from each experiment directory."""
    records = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        summary_path = exp_dir / "summary.json"
        config_path = exp_dir / "config.yaml"
        metadata_path = exp_dir / "metadata.json"

        if not summary_path.exists():
            continue

        with summary_path.open() as f:
            summary = json.load(f)

        record = {"experiment_id": exp_dir.name}

        # Extract metadata if available
        if metadata_path.exists():
            with metadata_path.open() as f:
                meta = json.load(f)
            record["policy_type"] = meta.get("policy_type", "unknown")
            record["num_agents"] = meta.get("num_agents", None)
            record["num_rounds"] = meta.get("num_rounds", None)
            record["seed"] = meta.get("seed", None)

        # Flatten summary metrics
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                record[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        record[f"{key}_{sub_key}"] = sub_value

        records.append(record)

    return records


def generate_condition_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Condition A (ablated/pure LLM) vs Condition B (grounded LLM)."""
    # Match both legacy naming (ablat/pure_llm/grounded) and the new
    # `_cond{A,B}` suffix produced by run_full_pipeline.py --condition.
    condition_a = df[df["experiment_id"].str.contains("ablat|pure_llm|_condA$", case=False, na=False, regex=True)]
    condition_b = df[df["experiment_id"].str.contains("grounded|_condB$", case=False, na=False, regex=True)]

    rows = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if col in ("seed", "num_agents", "num_rounds"):
            continue
        a_vals = condition_a[col].dropna()
        b_vals = condition_b[col].dropna()
        if a_vals.empty and b_vals.empty:
            continue
        rows.append(
            {
                "metric": col,
                "condition_a_mean": round(a_vals.mean(), 4) if not a_vals.empty else None,
                "condition_a_std": round(a_vals.std(), 4) if len(a_vals) > 1 else None,
                "condition_b_mean": round(b_vals.mean(), 4) if not b_vals.empty else None,
                "condition_b_std": round(b_vals.std(), 4) if len(b_vals) > 1 else None,
                "n_a": len(a_vals),
                "n_b": len(b_vals),
            }
        )

    return pd.DataFrame(rows)


def generate_policy_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by policy type."""
    if "policy_type" not in df.columns:
        return pd.DataFrame()

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c not in ("seed", "num_agents", "num_rounds")
    ]

    return df.groupby("policy_type")[numeric_cols].agg(["mean", "std", "count"]).round(4)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    records = load_experiment_summaries()
    if not records:
        print("No experiment summaries found.")
        return

    df = pd.DataFrame(records)

    # Table 1: All experiments overview
    overview_path = TABLES_DIR / "experiment_overview.csv"
    df.to_csv(overview_path, index=False)
    print(f"Wrote {overview_path} ({len(df)} experiments)")

    # Table 2: Condition A vs B comparison
    comparison = generate_condition_comparison(df)
    if not comparison.empty:
        cmp_path = TABLES_DIR / "condition_comparison.csv"
        comparison.to_csv(cmp_path, index=False)
        print(f"Wrote {cmp_path} ({len(comparison)} metrics)")

    # Table 3: Policy type comparison
    policy_cmp = generate_policy_comparison(df)
    if not policy_cmp.empty:
        policy_path = TABLES_DIR / "policy_comparison.csv"
        policy_cmp.to_csv(policy_path)
        print(f"Wrote {policy_path}")

    print("Done.")


if __name__ == "__main__":
    main()
