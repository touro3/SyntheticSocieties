"""ESS data validation script — integrity checks for parquet and distribution files.

Validates:
  1. ess_clean.parquet exists and is readable
  2. Expected columns are present (from ess_schema.py)
  3. No all-NaN columns
  4. Value ranges for normalized [0,1] columns
  5. Missing data rate per column
  6. dataset_registry.json is valid and references existing files
  7. empirical_distributions.json structure (if exists)

Usage:
    python scripts/validate_ess_data.py
    python scripts/validate_ess_data.py --data-dir data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")

# Columns expected to be normalized to [0, 1]
NORMALIZED_COLUMNS = [
    "trust_people", "trust_fairness", "trust_helpfulness",
    "trust_parliament", "trust_legal", "trust_police",
    "trust_politicians", "trust_parties", "trust_eu_parliament", "trust_un",
    "life_satisfaction", "happiness",
    "satisfaction_economy", "satisfaction_government",
    "satisfaction_democracy", "satisfaction_education",
    "satisfaction_health_sys",
    "left_right",
]

# Columns that should never be entirely NaN
REQUIRED_COLUMNS = [
    "country", "gender", "age",
    "trust_people", "life_satisfaction", "happiness",
]


class ValidationResult:
    """Accumulates pass/fail/warn checks."""

    def __init__(self):
        self.checks: list[dict] = []

    def ok(self, name: str, detail: str = ""):
        self.checks.append({"status": "PASS", "name": name, "detail": detail})

    def fail(self, name: str, detail: str = ""):
        self.checks.append({"status": "FAIL", "name": name, "detail": detail})

    def warn(self, name: str, detail: str = ""):
        self.checks.append({"status": "WARN", "name": name, "detail": detail})

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "PASS")

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "WARN")

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            symbol = {"PASS": "+", "FAIL": "X", "WARN": "!"}[c["status"]]
            line = f"  [{symbol}] {c['name']}"
            if c["detail"]:
                line += f" — {c['detail']}"
            lines.append(line)
        lines.append("")
        lines.append(f"  Total: {self.n_pass} passed, {self.n_fail} failed, {self.n_warn} warnings")
        return "\n".join(lines)


def validate_parquet(data_dir: Path, result: ValidationResult) -> pd.DataFrame | None:
    """Validate ess_clean.parquet exists and is readable."""
    path = data_dir / "ess_clean.parquet"

    if not path.exists():
        result.fail("parquet_exists", f"{path} not found")
        return None

    try:
        df = pd.read_parquet(path)
        result.ok("parquet_readable", f"{len(df)} rows x {len(df.columns)} columns")
    except Exception as e:
        result.fail("parquet_readable", str(e))
        return None

    if len(df) == 0:
        result.fail("parquet_nonempty", "DataFrame is empty")
        return None
    result.ok("parquet_nonempty", f"{len(df)} respondents")

    return df


def validate_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check required columns are present and not all-NaN."""
    cols = set(df.columns)

    for col in REQUIRED_COLUMNS:
        if col not in cols:
            result.fail(f"column_present:{col}", "missing from parquet")
        elif df[col].isna().all():
            result.fail(f"column_not_all_nan:{col}", "all values are NaN")
        else:
            result.ok(f"column_present:{col}")


def validate_ranges(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check normalized columns are within [0, 1]."""
    cols = set(df.columns)
    for col in NORMALIZED_COLUMNS:
        if col not in cols:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            result.warn(f"range:{col}", "no non-NaN values to check")
            continue
        vmin, vmax = series.min(), series.max()
        if vmin < -0.01 or vmax > 1.01:
            result.fail(f"range:{col}", f"min={vmin:.4f}, max={vmax:.4f} (expected [0,1])")
        else:
            result.ok(f"range:{col}", f"[{vmin:.3f}, {vmax:.3f}]")


def validate_missing_rates(df: pd.DataFrame, result: ValidationResult) -> None:
    """Warn if any column has >50% missing values."""
    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > 0.5:
            result.warn(f"missing_rate:{col}", f"{rate:.1%} missing")
        elif rate > 0.2:
            result.warn(f"missing_rate:{col}", f"{rate:.1%} missing (moderate)")


def validate_registry(data_dir: Path, result: ValidationResult) -> None:
    """Validate dataset_registry.json structure and file references."""
    path = data_dir / "dataset_registry.json"
    if not path.exists():
        result.warn("registry_exists", "dataset_registry.json not found")
        return

    try:
        with path.open() as f:
            registry = json.load(f)
        result.ok("registry_valid_json")
    except json.JSONDecodeError as e:
        result.fail("registry_valid_json", str(e))
        return

    datasets = registry.get("datasets", [])
    if not datasets:
        result.fail("registry_has_datasets", "no datasets found")
        return
    result.ok("registry_has_datasets", f"{len(datasets)} dataset(s)")

    for ds in datasets:
        ds_id = ds.get("id", "unknown")
        local_path = ds.get("local_path")
        if local_path:
            full = Path(local_path)
            if full.exists():
                result.ok(f"registry_file:{ds_id}", f"{local_path} exists")
            else:
                result.fail(f"registry_file:{ds_id}", f"{local_path} not found")

        # Check target_items structure
        items = ds.get("target_items", [])
        for item in items:
            if "name" not in item:
                result.fail(f"registry_item_schema:{ds_id}", "target_item missing 'name'")


def validate_distributions(data_dir: Path, result: ValidationResult) -> None:
    """Validate empirical_distributions.json if present."""
    path = data_dir / "empirical_distributions.json"
    if not path.exists():
        result.warn("distributions_exists", "empirical_distributions.json not found")
        return

    try:
        with path.open() as f:
            dist = json.load(f)
        result.ok("distributions_valid_json")
    except json.JSONDecodeError as e:
        result.fail("distributions_valid_json", str(e))
        return

    if not isinstance(dist, dict):
        result.fail("distributions_is_dict", f"expected dict, got {type(dist).__name__}")
        return

    result.ok("distributions_is_dict", f"{len(dist)} distribution(s)")

    for key, val in dist.items():
        if isinstance(val, dict):
            if "bins" in val and "counts" in val:
                if len(val["bins"]) != len(val["counts"]) + 1:
                    result.fail(f"dist_bins:{key}", "len(bins) != len(counts) + 1")
            elif "values" in val and "probabilities" in val:
                if len(val["values"]) != len(val["probabilities"]):
                    result.fail(f"dist_vals:{key}", "len(values) != len(probabilities)")
                prob_sum = sum(val["probabilities"])
                if abs(prob_sum - 1.0) > 0.01:
                    result.warn(f"dist_probs:{key}", f"probabilities sum to {prob_sum:.4f}")


def run_validation(data_dir: Path) -> ValidationResult:
    """Run all validation checks and return results."""
    result = ValidationResult()

    print(f"Validating ESS data in: {data_dir.resolve()}")
    print("=" * 60)

    df = validate_parquet(data_dir, result)
    if df is not None:
        validate_columns(df, result)
        validate_ranges(df, result)
        validate_missing_rates(df, result)

    validate_registry(data_dir, result)
    validate_distributions(data_dir, result)

    print(result.summary())
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="ESS data validation")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    args = parser.parse_args()

    result = run_validation(Path(args.data_dir))
    sys.exit(1 if result.n_fail > 0 else 0)


if __name__ == "__main__":
    main()
