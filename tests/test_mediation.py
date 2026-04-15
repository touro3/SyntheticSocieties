"""Tests for mediation analysis — decomposing grounding effects.

Phase 19 — Causal inference and ablation formalization.
"""

from __future__ import annotations

import pytest

from metrics.mediation import compute_mediation_decomposition, mediation_table


class TestMediationDecomposition:
    def test_total_equals_components(self):
        result = compute_mediation_decomposition(
            full_grounded_coop=0.35,
            persona_only_coop=0.50,
            rag_only_coop=0.55,
            baseline_coop=0.75,
        )
        total = result["total_effect"]
        persona = result["persona_effect"]
        rag = result["rag_effect"]
        interaction = result["interaction_effect"]
        assert total == pytest.approx(persona + rag + interaction, abs=1e-10)

    def test_identical_conditions_zero_effect(self):
        result = compute_mediation_decomposition(
            full_grounded_coop=0.50,
            persona_only_coop=0.50,
            rag_only_coop=0.50,
            baseline_coop=0.50,
        )
        assert result["total_effect"] == pytest.approx(0.0)
        assert result["persona_effect"] == pytest.approx(0.0)
        assert result["rag_effect"] == pytest.approx(0.0)
        assert result["interaction_effect"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        result = compute_mediation_decomposition(
            full_grounded_coop=0.30,
            persona_only_coop=0.50,
            rag_only_coop=0.60,
            baseline_coop=0.75,
        )
        expected_keys = {
            "total_effect",
            "persona_effect",
            "rag_effect",
            "interaction_effect",
            "persona_share",
            "rag_share",
        }
        assert set(result.keys()) == expected_keys

    def test_shares_sum_with_interaction(self):
        result = compute_mediation_decomposition(
            full_grounded_coop=0.30,
            persona_only_coop=0.50,
            rag_only_coop=0.55,
            baseline_coop=0.75,
        )
        # Shares are persona/total and rag/total — interaction fills the rest
        total = result["total_effect"]
        if abs(total) > 1e-10:
            reconstructed = result["persona_share"] * total + result["rag_share"] * total + result["interaction_effect"]
            assert reconstructed == pytest.approx(total, abs=1e-10)

    def test_negative_total_effect(self):
        """Grounding reduces cooperation (expected in BGF)."""
        result = compute_mediation_decomposition(
            full_grounded_coop=0.30,
            persona_only_coop=0.50,
            rag_only_coop=0.55,
            baseline_coop=0.75,
        )
        assert result["total_effect"] < 0

    def test_zero_total_shares_are_zero(self):
        result = compute_mediation_decomposition(
            full_grounded_coop=0.50,
            persona_only_coop=0.50,
            rag_only_coop=0.50,
            baseline_coop=0.50,
        )
        assert result["persona_share"] == 0.0
        assert result["rag_share"] == 0.0


class TestMediationTable:
    def test_returns_dataframe(self):
        conditions = {
            "baseline": {"coop_rate": 0.75, "gini": 0.05},
            "persona_only": {"coop_rate": 0.50, "gini": 0.15},
            "rag_only": {"coop_rate": 0.55, "gini": 0.10},
            "full_grounded": {"coop_rate": 0.30, "gini": 0.35},
        }
        df = mediation_table(conditions)
        assert "total_effect" in df.columns
        assert "persona_effect" in df.columns
        assert "rag_effect" in df.columns
        assert len(df) > 0

    def test_table_has_all_metrics(self):
        conditions = {
            "baseline": {"coop_rate": 0.75, "gini": 0.05, "brm": 0.30},
            "persona_only": {"coop_rate": 0.50, "gini": 0.15, "brm": 0.50},
            "rag_only": {"coop_rate": 0.55, "gini": 0.10, "brm": 0.45},
            "full_grounded": {"coop_rate": 0.30, "gini": 0.35, "brm": 0.80},
        }
        df = mediation_table(conditions)
        assert len(df) == 3  # One row per metric
