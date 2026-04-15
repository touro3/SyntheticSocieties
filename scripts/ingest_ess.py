"""
ESS Data Ingestion Pipeline for the Behavioral Grounding Framework.

Loads both ESS11 datasets (interview metadata + main survey data),
joins them, selects behaviorally relevant variables, handles missing
values, normalizes scales, and outputs clean Parquet + empirical
distributions.

Usage:
    python scripts/ingest_ess.py
    python scripts/ingest_ess.py --interview data/ESS11INTe04_1.csv --main data/ESS11MD_e01_2.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.ess_schema import (
    ALL_VARIABLE_GROUPS,
    INTERVIEW_META,
    get_ess_columns,
    get_rename_mapping,
)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_INTERVIEW = "data/ESS11INTe04_1.csv"
DEFAULT_MAIN = "data/ESS11MD_e01_2.csv"
OUTPUT_PARQUET = "data/ess_clean.parquet"
OUTPUT_BEHAVIOR = "data/behavior/ess_behavior_dataset.csv"
OUTPUT_DISTS = "data/empirical_distributions.json"


# ── Loading ──────────────────────────────────────────────────────────────────


def load_interview(path: str) -> pd.DataFrame:
    """Load the ESS interview metadata file."""
    df = pd.read_csv(path, low_memory=False)
    interview_cols = [c for c, _, _ in INTERVIEW_META]
    available = [c for c in interview_cols if c in df.columns]
    df = df[available].copy()
    print(f"Interview data loaded: {df.shape}")
    return df


def load_main(path: str) -> pd.DataFrame:
    """Load the ESS main data file, selecting only relevant columns."""
    target_cols = get_ess_columns()

    # Read just the header to find available columns
    header = pd.read_csv(path, nrows=0, low_memory=False)
    available = [c for c in target_cols if c in header.columns]
    missing = [c for c in target_cols if c not in header.columns]

    if missing:
        print(f"Warning: {len(missing)} target columns not found in main data: {missing[:10]}...")

    df = pd.read_csv(path, usecols=available, low_memory=False)
    print(f"Main data loaded: {df.shape} ({len(available)} of {len(target_cols)} target columns)")
    return df


# ── Joining ──────────────────────────────────────────────────────────────────


def join_datasets(interview: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Join interview and main data on respondent ID + country."""
    # Identify shared keys
    join_keys = []
    for key in ["idno", "cntry"]:
        if key in interview.columns and key in main.columns:
            join_keys.append(key)

    if not join_keys:
        print("Warning: No shared join keys found. Using main data only.")
        return main

    # Interview has interview-specific columns; merge them into main
    interview_only_cols = [c for c in interview.columns if c not in main.columns]
    if not interview_only_cols:
        print("No additional interview columns to merge. Using main data only.")
        return main

    merged = main.merge(
        interview[join_keys + interview_only_cols],
        on=join_keys,
        how="left",
    )
    print(f"Merged dataset: {merged.shape}")
    return merged


# ── Cleaning ─────────────────────────────────────────────────────────────────


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace ESS special codes with NaN.

    ESS uses numeric codes for missing/invalid responses. Multi-digit codes
    (66, 77, 88, 99 and their longer variants) are unambiguous missing codes
    for Likert-scale variables. Single-digit codes (6, 7, 8, 9) are NOT
    replaced because they overlap with valid responses on 0-10 scales.
    """
    # Only use multi-digit codes to avoid destroying valid Likert responses.
    # Single-digit codes (6, 7, 8, 9) are valid answers on 0-10 or 1-7 scales.
    multi_digit_missing = {
        55,
        66,
        77,
        88,
        99,
        555,
        666,
        777,
        888,
        999,
        5555,
        6666,
        7777,
        8888,
        9999,
        55555,
        66666,
        77777,
        88888,
        99999,
        555555,
        666666,
        777777,
        888888,
        999999,
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace(multi_digit_missing, np.nan)

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ESS variable codes to human-readable names."""
    mapping = get_rename_mapping()
    available_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=available_mapping)


def normalize_scales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ESS Likert scales to [0, 1] range.

    Many ESS variables use 0-10 scales (trust, satisfaction, happiness)
    and 1-4 or 1-5 scales (wellbeing, attitudes).
    """
    # 0–10 scale variables → [0, 1]
    scale_0_10 = [
        "trust_people",
        "trust_fairness",
        "trust_helpfulness",
        "trust_parliament",
        "trust_legal",
        "trust_police",
        "trust_politicians",
        "trust_parties",
        "trust_eu_parliament",
        "trust_un",
        "left_right",
        "life_satisfaction",
        "satisfaction_economy",
        "satisfaction_government",
        "satisfaction_democracy",
        "satisfaction_education",
        "satisfaction_health_sys",
        "happiness",
    ]
    for col in scale_0_10:
        if col in df.columns:
            df[col] = df[col] / 10.0

    # 0–6 scale variables → [0, 1]
    scale_0_6 = ["risk_taking", "leadership_preference", "competitiveness"]
    for col in scale_0_6:
        if col in df.columns:
            df[col] = df[col] / 6.0

    # 1–4 inverted scales (1=best, 4=worst → flip so higher = better) → [0, 1]
    scale_1_4_invert = [
        "immigration_same_ethnicity",
        "immigration_diff_ethnicity",
        "immigration_poor_countries",
        "feel_safe_dark",
    ]
    for col in scale_1_4_invert:
        if col in df.columns:
            df[col] = (4 - df[col]) / 3.0

    # 1–5 scales → [0, 1]
    scale_1_5 = [
        "reduce_inequality",
        "gay_rights",
    ]
    for col in scale_1_5:
        if col in df.columns:
            df[col] = (df[col] - 1) / 4.0

    # 1–7 scales → [0, 1]
    scale_1_7 = ["social_meeting_freq", "close_confidants"]
    for col in scale_1_7:
        if col in df.columns:
            df[col] = (df[col] - 1) / 6.0

    return df


# ── Distributions ────────────────────────────────────────────────────────────


def compute_distributions(df: pd.DataFrame) -> dict:
    """
    Compute empirical distributions for each variable group.
    Returns a JSON-serializable dict with value counts, means, and quantiles.
    """
    distributions = {"metadata": {"n_respondents": len(df), "n_variables": len(df.columns)}}

    for group_name, variables in ALL_VARIABLE_GROUPS.items():
        group_dists = {}
        for _, target_name, description in variables:
            if target_name not in df.columns:
                continue
            col = df[target_name].dropna()
            if len(col) == 0:
                continue

            entry = {"description": description, "n_valid": int(len(col))}

            if col.dtype in [np.float64, np.float32, float]:
                entry["type"] = "continuous"
                entry["mean"] = float(col.mean())
                entry["std"] = float(col.std())
                entry["min"] = float(col.min())
                entry["max"] = float(col.max())
                entry["quantiles"] = {str(q): float(col.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
            else:
                entry["type"] = "categorical"
                vc = col.value_counts(normalize=True)
                entry["value_counts"] = {str(k): float(v) for k, v in vc.items()}

            group_dists[target_name] = entry

        distributions[group_name] = group_dists

    return distributions


# ── Behavioral proxy ─────────────────────────────────────────────────────────


def generate_behavior_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate behavioral proxy actions from ESS survey responses.
    This creates the backward-compatible ess_behavior_dataset.csv output.

    Behavioral logic (richer than the original age/gender heuristic):
      - High trust + high social activity → cooperate
      - High risk + low satisfaction → work
      - Low risk + high satisfaction → save
      - Fallback → work
    """

    def classify_action(row):
        trust = row.get("trust_people", 0.5)
        social = row.get("social_meeting_freq", 0.5)
        risk = row.get("risk_taking", 0.5)
        satisfaction = row.get("life_satisfaction", 0.5)

        # Handle NaN: use midpoint
        if pd.isna(trust):
            trust = 0.5
        if pd.isna(social):
            social = 0.5
        if pd.isna(risk):
            risk = 0.5
        if pd.isna(satisfaction):
            satisfaction = 0.5

        if trust > 0.6 and social > 0.5:
            return "cooperate"
        if risk > 0.5 and satisfaction < 0.4:
            return "work"
        if risk < 0.4 and satisfaction > 0.6:
            return "save"
        return "work"

    df = df.copy()
    df["action"] = df.apply(classify_action, axis=1)
    return df


# ── Main pipeline ────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest ESS datasets for BGF.")
    parser.add_argument("--interview", default=DEFAULT_INTERVIEW, help="Path to interview CSV")
    parser.add_argument("--main", default=DEFAULT_MAIN, help="Path to main data CSV")
    parser.add_argument("--output", default=OUTPUT_PARQUET, help="Output Parquet path")
    parser.add_argument("--output-dists", default=OUTPUT_DISTS, help="Output distributions path")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("ESS Data Ingestion Pipeline — Behavioral Grounding Framework")
    print("=" * 60)

    # Step 1: Load datasets
    print("\n[1/6] Loading interview data...")
    interview = load_interview(args.interview)

    print("\n[2/6] Loading main data...")
    main_data = load_main(args.main)

    # Step 2: Join
    print("\n[3/6] Joining datasets...")
    merged = join_datasets(interview, main_data)

    # Step 3: Clean missing values
    print("\n[4/6] Cleaning missing values...")
    cleaned = clean_missing_values(merged)

    # Step 4: Rename to human-readable names
    cleaned = rename_columns(cleaned)

    # Step 5: Normalize scales
    cleaned = normalize_scales(cleaned)

    # Drop rows where all behavioral variables are NaN
    behavioral_cols = [c for c in cleaned.columns if c not in ["respondent_id", "country"]]
    cleaned = cleaned.dropna(subset=behavioral_cols, how="all")
    print(f"After cleaning: {cleaned.shape}")

    # Step 5: Save Parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    print(f"\n[5/6] Saved clean dataset: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Step 6: Compute and save distributions
    print("\n[6/6] Computing empirical distributions...")
    distributions = compute_distributions(cleaned)
    dist_path = Path(args.output_dists)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dist_path, "w") as f:
        json.dump(distributions, f, indent=2)
    print(f"Saved distributions: {dist_path}")

    # Also generate backward-compatible behavior dataset
    print("\nGenerating backward-compatible behavior dataset...")
    behavior_df = generate_behavior_proxy(cleaned)
    behavior_path = Path(OUTPUT_BEHAVIOR)
    behavior_path.parent.mkdir(parents=True, exist_ok=True)
    # Select a subset of columns for the behavior CSV
    behavior_cols = ["respondent_id", "country", "age", "gender"]
    behavior_cols = [c for c in behavior_cols if c in behavior_df.columns]
    behavior_cols.append("action")
    behavior_df[behavior_cols].to_csv(behavior_path, index=False)
    print(f"Saved behavior dataset: {behavior_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    print(f"Total respondents:  {len(cleaned)}")
    print(f"Total variables:    {len(cleaned.columns)}")
    print(f"Countries:          {cleaned['country'].nunique() if 'country' in cleaned.columns else 'N/A'}")

    if "country" in cleaned.columns:
        print("\nCountry breakdown:")
        for country, count in cleaned["country"].value_counts().items():
            print(f"  {country}: {count}")

    # Print action distribution
    if "action" in behavior_df.columns:
        print("\nBehavioral proxy distribution:")
        for action, count in behavior_df["action"].value_counts().items():
            pct = count / len(behavior_df) * 100
            print(f"  {action}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
