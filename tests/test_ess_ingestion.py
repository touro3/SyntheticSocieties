"""Tests for the ESS data ingestion pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def ess_clean_path():
    """Path to the cleaned ESS Parquet file."""
    path = Path("data/ess_clean.parquet")
    if not path.exists():
        pytest.skip("ESS clean data not found. Run: python scripts/ingest_ess.py")
    return str(path)


@pytest.fixture
def distributions_path():
    """Path to the empirical distributions JSON."""
    path = Path("data/empirical_distributions.json")
    if not path.exists():
        pytest.skip("Distributions file not found. Run: python scripts/ingest_ess.py")
    return str(path)


def test_ess_clean_parquet_exists_and_loads(ess_clean_path):
    import pandas as pd
    df = pd.read_parquet(ess_clean_path)
    assert len(df) > 0, "Dataset should have rows"
    assert len(df.columns) > 10, "Dataset should have many columns"


def test_ess_clean_has_expected_columns(ess_clean_path):
    import pandas as pd
    df = pd.read_parquet(ess_clean_path)
    expected = [
        "respondent_id", "country", "gender", "age",
        "trust_people", "life_satisfaction", "happiness",
    ]
    for col in expected:
        assert col in df.columns, f"Missing expected column: {col}"


def test_ess_clean_no_raw_missing_codes(ess_clean_path):
    """ESS missing codes (66, 77, 88, 99) should be replaced with NaN."""
    import pandas as pd
    import numpy as np
    df = pd.read_parquet(ess_clean_path)

    # Check normalized 0-1 columns — they should not have raw ESS codes
    normalized_cols = ["trust_people", "life_satisfaction", "happiness", "risk_taking"]
    for col in normalized_cols:
        if col in df.columns:
            valid = df[col].dropna()
            assert valid.max() <= 1.01, f"{col} has values > 1, possible missing code leak"
            assert valid.min() >= -0.01, f"{col} has values < 0"


def test_ess_distributions_json_structure(distributions_path):
    import json
    with open(distributions_path) as f:
        dists = json.load(f)

    assert "metadata" in dists
    assert dists["metadata"]["n_respondents"] > 0
    assert "demographics" in dists
    assert "trust" in dists
    assert "values_attitudes" in dists


def test_ess_schema_columns():
    from data.ess_schema import get_ess_columns, get_rename_mapping
    cols = get_ess_columns()
    assert "idno" in cols
    assert "cntry" in cols
    assert "ppltrst" in cols
    assert len(cols) > 20

    mapping = get_rename_mapping()
    assert mapping["ppltrst"] == "trust_people"
    assert mapping["agea"] == "age"


def test_ess_schema_variable_groups():
    from data.ess_schema import ALL_VARIABLE_GROUPS
    assert "demographics" in ALL_VARIABLE_GROUPS
    assert "trust" in ALL_VARIABLE_GROUPS
    assert "risk_personality" in ALL_VARIABLE_GROUPS
    assert len(ALL_VARIABLE_GROUPS) >= 7
