"""Tests for scripts/validate_ess_data.py — ESS data integrity checks."""

import json

import numpy as np
import pandas as pd

from scripts.validate_ess_data import (
    ValidationResult,
    run_validation,
    validate_columns,
    validate_distributions,
    validate_missing_rates,
    validate_parquet,
    validate_ranges,
    validate_registry,
)

# ── ValidationResult ─────────────────────────────────────────────────────────


class TestValidationResult:
    def test_counts(self):
        r = ValidationResult()
        r.ok("a")
        r.ok("b")
        r.fail("c")
        r.warn("d")
        assert r.n_pass == 2
        assert r.n_fail == 1
        assert r.n_warn == 1

    def test_summary_contains_all(self):
        r = ValidationResult()
        r.ok("check1", "fine")
        r.fail("check2", "broken")
        s = r.summary()
        assert "check1" in s
        assert "check2" in s
        assert "1 passed" in s
        assert "1 failed" in s


# ── validate_parquet ─────────────────────────────────────────────────────────


class TestValidateParquet:
    def test_missing_file(self, tmp_path):
        r = ValidationResult()
        df = validate_parquet(tmp_path, r)
        assert df is None
        assert r.n_fail >= 1

    def test_valid_parquet(self, tmp_path):
        df = pd.DataFrame({"country": ["FR", "DE"], "age": [30, 40]})
        df.to_parquet(tmp_path / "ess_clean.parquet")
        r = ValidationResult()
        result = validate_parquet(tmp_path, r)
        assert result is not None
        assert r.n_fail == 0
        assert len(result) == 2

    def test_empty_parquet(self, tmp_path):
        df = pd.DataFrame({"country": pd.Series([], dtype=str)})
        df.to_parquet(tmp_path / "ess_clean.parquet")
        r = ValidationResult()
        result = validate_parquet(tmp_path, r)
        assert result is None
        assert r.n_fail >= 1


# ── validate_columns ─────────────────────────────────────────────────────────


class TestValidateColumns:
    def test_all_present(self):
        df = pd.DataFrame(
            {
                "country": ["FR"],
                "gender": [1],
                "age": [30],
                "trust_people": [0.5],
                "life_satisfaction": [0.7],
                "happiness": [0.6],
            }
        )
        r = ValidationResult()
        validate_columns(df, r)
        assert r.n_fail == 0

    def test_missing_column(self):
        df = pd.DataFrame({"country": ["FR"], "age": [30]})
        r = ValidationResult()
        validate_columns(df, r)
        assert r.n_fail > 0  # gender, trust_people, etc. missing

    def test_all_nan_column(self):
        df = pd.DataFrame(
            {
                "country": ["FR"],
                "gender": [np.nan],
                "age": [30],
                "trust_people": [0.5],
                "life_satisfaction": [0.7],
                "happiness": [0.6],
            }
        )
        r = ValidationResult()
        validate_columns(df, r)
        assert any("gender" in c["name"] and c["status"] == "FAIL" for c in r.checks)


# ── validate_ranges ──────────────────────────────────────────────────────────


class TestValidateRanges:
    def test_valid_range(self):
        df = pd.DataFrame({"trust_people": [0.0, 0.5, 1.0]})
        r = ValidationResult()
        validate_ranges(df, r)
        assert r.n_fail == 0

    def test_out_of_range(self):
        df = pd.DataFrame({"trust_people": [0.0, 1.5]})
        r = ValidationResult()
        validate_ranges(df, r)
        assert r.n_fail == 1

    def test_all_nan_warns(self):
        df = pd.DataFrame({"trust_people": [np.nan, np.nan]})
        r = ValidationResult()
        validate_ranges(df, r)
        assert r.n_warn >= 1


# ── validate_missing_rates ───────────────────────────────────────────────────


class TestValidateMissingRates:
    def test_high_missing_warns(self):
        df = pd.DataFrame({"col": [np.nan] * 8 + [1.0] * 2})
        r = ValidationResult()
        validate_missing_rates(df, r)
        assert r.n_warn >= 1

    def test_low_missing_no_warn(self):
        df = pd.DataFrame({"col": [1.0] * 100})
        r = ValidationResult()
        validate_missing_rates(df, r)
        assert r.n_warn == 0


# ── validate_registry ────────────────────────────────────────────────────────


class TestValidateRegistry:
    def test_missing_registry_warns(self, tmp_path):
        r = ValidationResult()
        validate_registry(tmp_path, r)
        assert r.n_warn >= 1

    def test_valid_registry(self, tmp_path):
        registry = {
            "datasets": [
                {
                    "id": "test",
                    "local_path": str(tmp_path / "ess_clean.parquet"),
                    "target_items": [{"name": "trust_people"}],
                }
            ]
        }
        (tmp_path / "ess_clean.parquet").write_bytes(b"")  # dummy file
        (tmp_path / "dataset_registry.json").write_text(json.dumps(registry))
        r = ValidationResult()
        validate_registry(tmp_path, r)
        assert r.n_fail == 0

    def test_missing_local_file_fails(self, tmp_path):
        registry = {"datasets": [{"id": "test", "local_path": str(tmp_path / "missing.parquet"), "target_items": []}]}
        (tmp_path / "dataset_registry.json").write_text(json.dumps(registry))
        r = ValidationResult()
        validate_registry(tmp_path, r)
        assert any("registry_file" in c["name"] and c["status"] == "FAIL" for c in r.checks)


# ── validate_distributions ───────────────────────────────────────────────────


class TestValidateDistributions:
    def test_missing_file_warns(self, tmp_path):
        r = ValidationResult()
        validate_distributions(tmp_path, r)
        assert r.n_warn >= 1

    def test_valid_distribution(self, tmp_path):
        dist = {"age": {"values": [20, 30, 40], "probabilities": [0.3, 0.4, 0.3]}}
        (tmp_path / "empirical_distributions.json").write_text(json.dumps(dist))
        r = ValidationResult()
        validate_distributions(tmp_path, r)
        assert r.n_fail == 0

    def test_probability_sum_mismatch(self, tmp_path):
        dist = {"age": {"values": [20, 30], "probabilities": [0.3, 0.3]}}
        (tmp_path / "empirical_distributions.json").write_text(json.dumps(dist))
        r = ValidationResult()
        validate_distributions(tmp_path, r)
        assert r.n_warn >= 1  # probs sum to 0.6


# ── run_validation (integration) ─────────────────────────────────────────────


class TestRunValidation:
    def test_full_valid_run(self, tmp_path):
        df = pd.DataFrame(
            {
                "country": ["FR", "DE", "UK"],
                "gender": [1, 2, 1],
                "age": [30, 40, 50],
                "trust_people": [0.5, 0.7, 0.3],
                "life_satisfaction": [0.6, 0.8, 0.4],
                "happiness": [0.7, 0.9, 0.5],
            }
        )
        df.to_parquet(tmp_path / "ess_clean.parquet")
        result = run_validation(tmp_path)
        assert result.n_fail == 0
