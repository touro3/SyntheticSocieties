"""Tests for metrics/calibration.py — calibration/evaluation split, metric computation."""

import json
from pathlib import Path

import numpy as np
import pytest

from metrics.calibration import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    POLICY_PREFIX,
    calibration_evaluation_split,
    calibration_report,
    compute_metrics,
    load_summary,
)


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_calibration_seeds_are_list(self):
        assert isinstance(CALIBRATION_SEEDS, list)
        assert len(CALIBRATION_SEEDS) >= 1

    def test_evaluation_seeds_are_list(self):
        assert isinstance(EVALUATION_SEEDS, list)
        assert len(EVALUATION_SEEDS) >= 1

    def test_no_seed_overlap(self):
        assert set(CALIBRATION_SEEDS).isdisjoint(set(EVALUATION_SEEDS))

    def test_policy_prefix_contains_expected_policies(self):
        for p in ["llm", "template", "rule_based", "random"]:
            assert p in POLICY_PREFIX


# ── load_summary ─────────────────────────────────────────────────────────────


class TestLoadSummary:
    def test_missing_experiment_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        result = load_summary("nonexistent_exp")
        assert result == {}

    def test_loads_valid_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        exp_dir = tmp_path / "test_exp"
        exp_dir.mkdir()
        data = {"wealth": {"values": [10, 20, 30]}, "event_action_counts": {"work": 5}}
        (exp_dir / "summary.json").write_text(json.dumps(data))
        result = load_summary("test_exp")
        assert result["wealth"]["values"] == [10, 20, 30]


# ── compute_metrics ──────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_empty_experiment_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        result = compute_metrics([])
        assert result["mean_wealth"] == 0.0
        assert result["gini"] == 0.0
        assert result["coop_rate"] == 0.0
        assert result["n_agents"] == 0

    def test_nonexistent_experiments(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        result = compute_metrics(["ghost_exp1", "ghost_exp2"])
        assert result["n_agents"] == 0

    def test_valid_experiments(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)

        # Create two experiment summaries
        for exp_id, wealth, actions in [
            ("exp1", [100, 100, 100], {"work": 3}),
            ("exp2", [50, 150], {"cooperate": 2, "work": 1}),
        ]:
            d = tmp_path / exp_id
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({
                "wealth": {"values": wealth},
                "event_action_counts": actions,
            }))

        result = compute_metrics(["exp1", "exp2"])
        assert result["n_agents"] == 5
        assert result["mean_wealth"] == pytest.approx(100.0)
        assert result["coop_rate"] == pytest.approx(2 / 6)
        assert "gini" in result
        assert "std_wealth" in result
        assert "action_counts" in result

    def test_uniform_wealth_has_zero_gini(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        d = tmp_path / "uniform"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({
            "wealth": {"values": [50, 50, 50, 50]},
            "event_action_counts": {"work": 4},
        }))
        result = compute_metrics(["uniform"])
        assert result["gini"] == pytest.approx(0.0)

    def test_single_agent_gini_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        d = tmp_path / "single"
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({
            "wealth": {"values": [100]},
            "event_action_counts": {"work": 1},
        }))
        result = compute_metrics(["single"])
        assert result["gini"] == 0.0


# ── calibration_evaluation_split ─────────────────────────────────────────────


class TestCalibrationEvaluationSplit:
    def test_returns_expected_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        result = calibration_evaluation_split("llm")
        assert "policy" in result
        assert "calibration" in result
        assert "evaluation" in result
        assert "gap" in result
        assert result["policy"] == "llm"

    def test_gap_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        result = calibration_evaluation_split("random")
        for key in ["mean_wealth_gap_pct", "gini_gap_pct", "coop_rate_gap_pct"]:
            assert key in result["gap"]

    def test_zero_gap_when_no_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        result = calibration_evaluation_split("template")
        for val in result["gap"].values():
            assert val == 0.0

    def test_gap_calculation_with_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        # Create calibration experiment
        for seed in CALIBRATION_SEEDS:
            d = tmp_path / f"cmp_llm_s{seed}"
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({
                "wealth": {"values": [100, 100]},
                "event_action_counts": {"work": 2},
            }))
        # Create evaluation experiment
        for seed in EVALUATION_SEEDS:
            d = tmp_path / f"cmp_llm_s{seed}"
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({
                "wealth": {"values": [80, 80]},
                "event_action_counts": {"cooperate": 2},
            }))

        result = calibration_evaluation_split("llm")
        assert result["calibration"]["mean_wealth"] == pytest.approx(100.0)
        assert result["evaluation"]["mean_wealth"] == pytest.approx(80.0)
        assert result["gap"]["mean_wealth_gap_pct"] == pytest.approx(20.0)


# ── calibration_report ───────────────────────────────────────────────────────


class TestCalibrationReport:
    def test_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        report = calibration_report(policies=["random"])
        assert isinstance(report, str)
        assert "Calibration vs Evaluation Report" in report

    def test_contains_policy_labels(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        report = calibration_report(policies=["llm", "random"])
        assert "LLM (Mistral)" in report or "N/A" in report

    def test_no_data_shows_na(self, tmp_path, monkeypatch):
        monkeypatch.setattr("metrics.calibration.EXPERIMENTS_DIR", tmp_path)
        report = calibration_report(policies=["template"])
        assert "N/A" in report or "(no data)" in report


# ── compute_metrics edge cases ────────────────────────────────────────────────


class TestComputeMetricsEdgeCases:
    def test_empty_experiment_list_returns_zero_sentinel(self):
        # compute_metrics([]) must return the zero-sentinel dict, never raise.
        result = compute_metrics([])
        assert isinstance(result, dict)
        assert result.get("n_agents", 0) == 0
