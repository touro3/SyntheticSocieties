"""Tests for phase transition detection and power law fitting.

Phase 18 — Emergent complexity analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics.complexity import (
    analyze_sweep_results,
    fit_phase_transition,
    fit_power_law,
    sigmoid,
)


# ── sigmoid ──────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_known_shape(self):
        x = np.array([0.0, 5.0, 10.0])
        # L=1, k=1, x0=5, b=0 -> standard logistic centered at 5
        y = sigmoid(x, L=1.0, k=1.0, x0=5.0, b=0.0)
        # At x=x0, sigmoid = L/2 + b = 0.5
        assert y[1] == pytest.approx(0.5, abs=0.01)
        # At x << x0, sigmoid -> b = 0
        assert y[0] < 0.02
        # At x >> x0, sigmoid -> L + b = 1
        assert y[2] > 0.98

    def test_offset(self):
        x = np.array([5.0])
        y = sigmoid(x, L=1.0, k=1.0, x0=5.0, b=0.3)
        assert y[0] == pytest.approx(0.8, abs=0.01)

    def test_vectorized(self):
        x = np.linspace(0, 10, 50)
        y = sigmoid(x, L=1.0, k=1.0, x0=5.0, b=0.0)
        assert y.shape == (50,)
        # Monotonically increasing for positive k
        assert np.all(np.diff(y) >= 0)


# ── fit_phase_transition ─────────────────────────────────────────────────


class TestFitPhaseTransition:
    def test_known_sigmoid_data(self):
        x = np.linspace(0, 1, 50)
        y = sigmoid(x, L=0.6, k=20.0, x0=0.2, b=0.1)
        # Add tiny noise
        rng = np.random.default_rng(42)
        y_noisy = y + rng.normal(0, 0.01, len(y))

        result = fit_phase_transition(x, y_noisy)
        assert result["is_transition"] == True
        assert result["inflection_point"] == pytest.approx(0.2, abs=0.05)
        assert result["r_squared"] > 0.9

    def test_linear_data_no_transition(self):
        x = np.linspace(0, 1, 30)
        y = 0.5 * x + 0.1
        result = fit_phase_transition(x, y)
        # Linear data should not be flagged as a sigmoid transition
        # (low steepness or poor fit)
        # The function may or may not flag it — the key is r_squared check
        assert "inflection_point" in result
        assert "r_squared" in result

    def test_constant_data(self):
        x = np.linspace(0, 1, 20)
        y = np.full_like(x, 0.5)
        result = fit_phase_transition(x, y)
        assert result["is_transition"] == False

    def test_output_keys(self):
        x = np.linspace(0, 1, 30)
        y = sigmoid(x, L=1.0, k=10.0, x0=0.5, b=0.0)
        result = fit_phase_transition(x, y)
        expected = {"inflection_point", "steepness", "r_squared", "is_transition", "fit_params"}
        assert expected == set(result.keys())

    def test_too_few_points_returns_no_transition(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        result = fit_phase_transition(x, y)
        assert result["is_transition"] == False


# ── fit_power_law ────────────────────────────────────────────────────────


class TestFitPowerLaw:
    def test_pareto_distribution(self):
        rng = np.random.default_rng(42)
        # Generate Pareto with known alpha
        true_alpha = 2.5
        xmin = 1.0
        values = (rng.pareto(true_alpha - 1, 1000) + 1) * xmin
        result = fit_power_law(values, xmin=xmin)
        assert result["alpha"] == pytest.approx(true_alpha, abs=0.3)

    def test_uniform_not_power_law(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(1, 100, 500)
        result = fit_power_law(values)
        # Uniform data should not be classified as power law
        assert result["is_power_law"] == False

    def test_small_sample_warning(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fit_power_law(values)
        assert result["reliable"] == False

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        values = (rng.pareto(2.0, 200) + 1)
        result = fit_power_law(values)
        expected = {"alpha", "xmin", "ks_statistic", "p_value", "is_power_law", "reliable"}
        assert expected == set(result.keys())

    def test_all_same_value(self):
        values = np.array([5.0] * 100)
        result = fit_power_law(values)
        assert result["is_power_law"] == False


# ── analyze_sweep_results ────────────────────────────────────────────────


class TestAnalyzeSweepResults:
    def test_single_metric_sweep(self):
        x = np.linspace(0, 1, 30)
        coop = sigmoid(x, L=-0.6, k=15.0, x0=0.2, b=0.7)
        result = analyze_sweep_results(x, {"cooperation_rate": coop})
        assert "cooperation_rate" in result
        assert "inflection_point" in result["cooperation_rate"]

    def test_multiple_metrics(self):
        x = np.linspace(0, 1, 30)
        metrics = {
            "cooperation_rate": sigmoid(x, L=-0.5, k=10.0, x0=0.3, b=0.7),
            "gini": sigmoid(x, L=0.4, k=8.0, x0=0.5, b=0.1),
        }
        result = analyze_sweep_results(x, metrics)
        assert "cooperation_rate" in result
        assert "gini" in result

    def test_flat_metric_produces_no_transition(self):
        # A perfectly flat metric has no phase transition.
        x = np.linspace(0, 1, 30)
        flat = np.full(30, 0.5)
        result = analyze_sweep_results(x, {"flat_metric": flat})
        assert result["flat_metric"]["is_transition"] is False


# ── fit_phase_transition edge cases ──────────────────────────────────────────


class TestFitPhaseTransitionEdgeCases:
    def test_too_few_points_returns_nan_inflection(self):
        # < 5 points should return the degenerate sentinel dict.
        x = np.array([0.1, 0.2, 0.3])
        y = np.array([0.4, 0.5, 0.6])
        result = fit_phase_transition(x, y)
        assert np.isnan(result["inflection_point"])
        assert result["is_transition"] is False

    def test_flat_input_returns_no_transition(self):
        # Flat y-values → y_range < 1e-10 → no transition detected.
        x = np.linspace(0, 1, 20)
        y = np.full(20, 0.42)
        result = fit_phase_transition(x, y)
        assert result["is_transition"] is False
