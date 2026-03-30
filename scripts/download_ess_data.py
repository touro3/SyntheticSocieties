"""
download_ess_data.py — ESS Round 11 ingestion and cleaning script.

Usage:
    python scripts/download_ess_data.py --input data/raw/ESS11.sav --output data/ess_clean.parquet

The ESS Round 11 microdata must be downloaded manually from:
    https://ess.sikt.no/
Obtain a free academic licence, download the integrated SPSS or Stata file, and pass
it as --input.  This script converts it to a clean Parquet file that all BGF modules read.

Requirements (install via pip):
    pyreadstat   # for .sav / .dta ingestion
    pandas
    pyarrow      # for Parquet output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Variable schema ──────────────────────────────────────────────────────────
# Maps ESS column names → BGF attribute names and specifies normalization.
#
# Scale variables are normalised to [0, 1] from their raw integer range.
# Binary variables are kept as 0/1 integers.
# Income is recoded from the 10-decile ordinal to a [0, 1] continuous score.

ESS_TO_BGF = {
    # Trust
    "ppltrst":  ("trust_people",        0, 10),   # 0=no trust … 10=complete trust
    "trstprl":  ("trust_parliament",    0, 10),
    "trstlgl":  ("trust_legal",         0, 10),
    # Wellbeing / satisfaction
    "stflife":  ("life_satisfaction",   0, 10),
    "stfeco":   ("econ_satisfaction",   0, 10),
    # Economic
    "hinctnta": ("income_decile",       1, 10),   # 1-10 decile → normalised
    # Risk / cooperation
    "ipfrule":  ("rule_following",      1, 6),    # ESS Human Values item
    "ipeqopt":  ("equality_orientation", 1, 6),
    # Demographics
    "agea":     ("age",                 15, 90),
    "gndr":     ("gender",              1,  2),   # 1=male, 2=female
    "eduyrs":   ("education_years",     0, 25),
    # Social activity
    "sclmeet":  ("social_activity",     1,  7),   # 1=never … 7=every day
    # Country
    "cntry":    ("country",             None, None),  # string, kept as-is
}

# Variables that must be present (drop rows missing any of these)
REQUIRED_VARS = [
    "trust_people", "life_satisfaction", "income_decile",
    "age", "gender", "country",
]

# Competitiveness proxy: high equality orientation (reversed) × low rule-following
# Synthesised after normalisation.

COUNTRY_CLUSTERS = {
    "nordic":   ["NO", "SE", "DK"],
    "southern": ["IT", "ES", "PT"],
    "eastern":  ["PL", "CZ", "HU"],
}


def _normalise(series, lo, hi):
    """Clip to [lo, hi] then scale to [0, 1]."""
    import pandas as pd
    s = pd.to_numeric(series, errors="coerce").clip(lo, hi)
    return (s - lo) / (hi - lo)


def ingest(input_path: Path) -> "pd.DataFrame":
    """Load raw ESS file (SPSS or Stata) and return a raw DataFrame."""
    suffix = input_path.suffix.lower()
    try:
        import pyreadstat
    except ImportError:
        sys.exit(
            "ERROR: pyreadstat is required.\n"
            "Install with:  pip install pyreadstat pyarrow"
        )

    print(f"Loading {input_path} …", end=" ", flush=True)
    if suffix == ".sav":
        df, _ = pyreadstat.read_sav(str(input_path))
    elif suffix in {".dta", ".stata"}:
        df, _ = pyreadstat.read_dta(str(input_path))
    else:
        sys.exit(f"ERROR: Unsupported file type '{suffix}'. Provide .sav or .dta")

    print(f"done. {len(df):,} respondents, {len(df.columns)} variables.")
    return df


def clean(raw: "pd.DataFrame") -> "pd.DataFrame":
    """Recode, normalise, and filter the raw ESS DataFrame."""
    import pandas as pd

    bgf: dict = {}

    for ess_col, (bgf_col, lo, hi) in ESS_TO_BGF.items():
        if ess_col not in raw.columns:
            print(f"  WARNING: ESS column '{ess_col}' not found — filling with NaN.")
            bgf[bgf_col] = float("nan")
            continue

        if lo is None:
            # String / categorical (country code)
            bgf[bgf_col] = raw[ess_col].astype(str).str.strip().str.upper()
        else:
            bgf[bgf_col] = _normalise(raw[ess_col], lo, hi)

    out = pd.DataFrame(bgf)

    # Synthesise derived attributes
    if "rule_following" in out.columns and "equality_orientation" in out.columns:
        out["competitiveness"] = (1 - out["equality_orientation"]) * (1 - out["rule_following"])

    # Assign cluster label
    reverse_cluster: dict[str, str] = {
        country: cluster
        for cluster, countries in COUNTRY_CLUSTERS.items()
        for country in countries
    }
    out["cluster"] = out["country"].map(reverse_cluster).fillna("other")

    # Drop rows missing required variables
    before = len(out)
    out = out.dropna(subset=REQUIRED_VARS)
    dropped = before - len(out)
    print(f"  Dropped {dropped:,} rows with missing required variables.")
    print(f"  Clean dataset: {len(out):,} respondents.")

    # Country distribution summary
    country_counts = out["country"].value_counts()
    print("\n  Country distribution (top 10):")
    for country, count in country_counts.head(10).items():
        print(f"    {country}: {count:,}")

    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ESS Round 11 data and write BGF-clean Parquet."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to raw ESS file (.sav or .dta).",
    )
    parser.add_argument(
        "--output", default="data/ess_clean.parquet", type=Path,
        help="Output Parquet path (default: data/ess_clean.parquet).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(
            f"ERROR: Input file not found: {args.input}\n\n"
            "Download ESS Round 11 from https://ess.sikt.no/ and re-run.\n"
            "See data/README.md for step-by-step instructions."
        )

    raw = ingest(args.input)
    clean_df = clean(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(args.output, index=False)
    print(f"\nSaved clean dataset to {args.output} ({args.output.stat().st_size // 1024:,} KB).")


if __name__ == "__main__":
    main()
