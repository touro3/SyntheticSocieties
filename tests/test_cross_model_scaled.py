"""Tests for scaled cross-model analysis — Section 3 of TOP_TIER_RESEARCH.md.

Phase 29.3 — GPT-4o-mini N=50 inverse effect robustness.

Covers:
  - bootstrap_ci()  — percentile bootstrap confidence intervals
  - compute_inverse_effect_significance()  — is GPT-4o-mini inverse real at N=50?
  - aggregate_seeded_results()  — pool multi-seed runs per model+condition
  - build_scaled_comparison_table()  — N=20 + N=50 combined with CIs
  - cross-model config YAML files exist with correct scaled parameters
  - run_cross_model_comparison --seeds flag is accepted (dry-run)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_result(model_id, condition, b_rlhf, coop_rate=0.4, gini=0.3,
                 n_agents=50, n_rounds=20):
    return {
        "model_id": model_id,
        "condition": condition,
        "rlhf_bias_index": b_rlhf,
        "cooperation_rate": coop_rate,
        "gini": gini,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
    }


# ── bootstrap_ci ─────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_lower_upper(self):
        from scripts.analyze_cross_model_scaled import bootstrap_ci
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        lo, hi = bootstrap_ci(data, n_bootstrap=500, ci=0.95, seed=42)
        assert lo < hi

    def test_ci_contains_mean_for_symmetric_data(self):
        from scripts.analyze_cross_model_scaled import bootstrap_ci
        import statistics
        data = [0.3, 0.35, 0.32, 0.33, 0.31, 0.34]
        lo, hi = bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=7)
        mean = statistics.mean(data)
        assert lo <= mean <= hi

    def test_wider_with_more_variance(self):
        from scripts.analyze_cross_model_scaled import bootstrap_ci
        tight = [0.5, 0.5, 0.5, 0.5, 0.5]
        spread = [0.1, 0.3, 0.5, 0.7, 0.9]
        lo_t, hi_t = bootstrap_ci(tight, n_bootstrap=500, seed=42)
        lo_s, hi_s = bootstrap_ci(spread, n_bootstrap=500, seed=42)
        assert (hi_s - lo_s) > (hi_t - lo_t)

    def test_single_element_returns_equal_bounds(self):
        from scripts.analyze_cross_model_scaled import bootstrap_ci
        lo, hi = bootstrap_ci([0.42], n_bootstrap=100, seed=0)
        assert lo == pytest.approx(hi)

    def test_empty_raises(self):
        from scripts.analyze_cross_model_scaled import bootstrap_ci
        with pytest.raises((ValueError, IndexError)):
            bootstrap_ci([], n_bootstrap=100)


# ── aggregate_seeded_results ─────────────────────────────────────────────────

class TestAggregateSeededResults:
    def test_returns_mean_per_model_condition(self):
        from scripts.analyze_cross_model_scaled import aggregate_seeded_results
        rows = [
            _make_result("gpt4o-mini", "A", 0.2),
            _make_result("gpt4o-mini", "A", 0.3),
            _make_result("gpt4o-mini", "A", 0.25),
        ]
        agg = aggregate_seeded_results(rows)
        key = ("gpt4o-mini", "A")
        assert key in agg
        assert abs(agg[key]["b_rlhf_mean"] - 0.25) < 1e-9

    def test_stores_raw_values_for_ci(self):
        from scripts.analyze_cross_model_scaled import aggregate_seeded_results
        rows = [
            _make_result("mistral-7b", "B", 0.15),
            _make_result("mistral-7b", "B", 0.18),
        ]
        agg = aggregate_seeded_results(rows)
        assert len(agg[("mistral-7b", "B")]["b_rlhf_values"]) == 2

    def test_handles_multiple_models(self):
        from scripts.analyze_cross_model_scaled import aggregate_seeded_results
        rows = [
            _make_result("mistral-7b", "A", 0.5),
            _make_result("qwen2.5-7b", "A", 0.3),
        ]
        agg = aggregate_seeded_results(rows)
        assert ("mistral-7b", "A") in agg
        assert ("qwen2.5-7b", "A") in agg


# ── compute_inverse_effect_significance ──────────────────────────────────────

class TestInverseEffectSignificance:
    def test_detects_significant_inverse(self):
        """If B_RLHF_B > B_RLHF_A clearly across seeds, flag as significant."""
        from scripts.analyze_cross_model_scaled import compute_inverse_effect_significance

        # GPT-4o-mini inverse: B has HIGHER bias than A
        b_rlhf_a = [0.20, 0.22, 0.21]  # Condition A (lower bias)
        b_rlhf_b = [0.40, 0.42, 0.41]  # Condition B (higher bias — inverse!)
        result = compute_inverse_effect_significance(b_rlhf_a, b_rlhf_b)

        assert result["inverse_detected"] is True
        assert result["delta_mean"] > 0  # B > A (bias goes up with grounding)

    def test_no_inverse_when_b_lower(self):
        """Normal case: B_RLHF_B < B_RLHF_A — grounding works, no inverse."""
        from scripts.analyze_cross_model_scaled import compute_inverse_effect_significance

        b_rlhf_a = [0.50, 0.52, 0.48]
        b_rlhf_b = [0.20, 0.22, 0.21]
        result = compute_inverse_effect_significance(b_rlhf_a, b_rlhf_b)

        assert result["inverse_detected"] is False

    def test_returns_required_keys(self):
        from scripts.analyze_cross_model_scaled import compute_inverse_effect_significance

        result = compute_inverse_effect_significance([0.3, 0.3], [0.4, 0.4])
        for key in ("inverse_detected", "delta_mean", "p_value", "effect_size_d"):
            assert key in result

    def test_insufficient_data_no_crash(self):
        """Single-seed data (no repeated measures) should return gracefully."""
        from scripts.analyze_cross_model_scaled import compute_inverse_effect_significance
        result = compute_inverse_effect_significance([0.3], [0.4])
        assert "inverse_detected" in result


# ── build_scaled_comparison_table ────────────────────────────────────────────

class TestBuildScaledComparisonTable:
    def _n20_results(self):
        return [
            _make_result("mistral-7b", "A", 0.567, n_agents=20, n_rounds=10),
            _make_result("mistral-7b", "B", 0.467, n_agents=20, n_rounds=10),
            _make_result("gpt4o-mini", "A", 0.223, n_agents=20, n_rounds=10),
            _make_result("gpt4o-mini", "B", 0.313, n_agents=20, n_rounds=10),
        ]

    def _n50_results(self):
        return [
            _make_result("mistral-7b", "A", 0.55, n_agents=50, n_rounds=20),
            _make_result("mistral-7b", "A", 0.57, n_agents=50, n_rounds=20),
            _make_result("mistral-7b", "B", 0.46, n_agents=50, n_rounds=20),
            _make_result("mistral-7b", "B", 0.45, n_agents=50, n_rounds=20),
            _make_result("gpt4o-mini", "A", 0.22, n_agents=50, n_rounds=20),
            _make_result("gpt4o-mini", "A", 0.24, n_agents=50, n_rounds=20),
            _make_result("gpt4o-mini", "B", 0.32, n_agents=50, n_rounds=20),
            _make_result("gpt4o-mini", "B", 0.30, n_agents=50, n_rounds=20),
        ]

    def test_returns_list_of_rows(self):
        from scripts.analyze_cross_model_scaled import build_scaled_comparison_table
        table = build_scaled_comparison_table(
            n20_results=self._n20_results(),
            n50_results=self._n50_results(),
        )
        assert isinstance(table, list)
        assert len(table) > 0

    def test_row_has_required_keys(self):
        from scripts.analyze_cross_model_scaled import build_scaled_comparison_table
        table = build_scaled_comparison_table(
            n20_results=self._n20_results(),
            n50_results=self._n50_results(),
        )
        row = next(r for r in table if r["model"] == "gpt4o-mini")
        for key in ("model", "n20_bias_A", "n20_bias_B", "n50_bias_A_mean",
                    "n50_bias_B_mean", "inverse_effect"):
            assert key in row, f"Missing key: {key}"

    def test_gpt4o_mini_inverse_flagged(self):
        """GPT-4o-mini inverse effect should be flagged when B_RLHF_B > B_RLHF_A."""
        from scripts.analyze_cross_model_scaled import build_scaled_comparison_table
        table = build_scaled_comparison_table(
            n20_results=self._n20_results(),
            n50_results=self._n50_results(),
        )
        gpt_row = next(r for r in table if r["model"] == "gpt4o-mini")
        assert gpt_row["inverse_effect"] is True

    def test_mistral_not_inverse(self):
        """Mistral (grounding works) should not be flagged as inverse."""
        from scripts.analyze_cross_model_scaled import build_scaled_comparison_table
        table = build_scaled_comparison_table(
            n20_results=self._n20_results(),
            n50_results=self._n50_results(),
        )
        mistral_row = next(r for r in table if r["model"] == "mistral-7b")
        assert mistral_row["inverse_effect"] is False

    def test_n50_ci_bounds_present(self):
        from scripts.analyze_cross_model_scaled import build_scaled_comparison_table
        table = build_scaled_comparison_table(
            n20_results=self._n20_results(),
            n50_results=self._n50_results(),
        )
        for row in table:
            assert "n50_bias_A_ci_lo" in row
            assert "n50_bias_A_ci_hi" in row
            assert "n50_bias_B_ci_lo" in row
            assert "n50_bias_B_ci_hi" in row


# ── Config YAML files reflect N=50 / T=20 ────────────────────────────────────

class TestScaledConfigFiles:
    def test_gpt4o_mini_config_has_n50(self):
        import yaml
        path = Path("configs/cross_model/gpt4o_mini.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["simulation"]["population_size"] == 50

    def test_gpt4o_mini_config_has_t20(self):
        import yaml
        path = Path("configs/cross_model/gpt4o_mini.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["simulation"]["rounds"] == 20

    def test_qwen_config_has_n50(self):
        import yaml
        path = Path("configs/cross_model/qwen2.5-7b.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["simulation"]["population_size"] == 50

    def test_mistral_config_has_n50(self):
        import yaml
        path = Path("configs/cross_model/mistral.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["simulation"]["population_size"] == 50

    def test_mistral_config_has_t20(self):
        import yaml
        path = Path("configs/cross_model/mistral.yaml")
        cfg = yaml.safe_load(path.read_text())
        assert cfg["simulation"]["rounds"] == 20


# ── run_cross_model_comparison --seeds flag ───────────────────────────────────

class TestRunScriptSeedsFlag:
    def test_dry_run_with_seeds_exits_cleanly(self):
        """--seeds flag with --dry-run should complete without error."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/run_cross_model_comparison.py",
             "--dry-run", "--models", "mistral-7b", "--seeds", "42,123"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    def test_dry_run_produces_multi_seed_output(self):
        """With 3 seeds, dry-run should show 3 entries per model+condition."""
        import subprocess, sys, tempfile, json
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        result = subprocess.run(
            [sys.executable, "scripts/run_cross_model_comparison.py",
             "--dry-run", "--models", "mistral-7b",
             "--seeds", "42,123,7", "--out", out_path],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        saved = json.loads(Path(out_path).read_text())
        # 3 seeds × 2 conditions = 6 rows for mistral-7b
        mistral_rows = [r for r in saved if r["model_id"] == "mistral-7b"]
        assert len(mistral_rows) == 6
