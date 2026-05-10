"""Tests for social dilemmas module and standalone B_RLHF metric.

Tests the Universal Multi-Agent Misalignment Thesis infrastructure:
  - All game types instantiate and compute correctly
  - B_RLHF is monotone with cooperative concentration
  - Grounding effect correctly classifies direction
  - Cross-game runner produces consistent output
  - Thesis validation correctly classifies full vs partial support
"""

from __future__ import annotations

import pytest

from environment.social_dilemmas import (
    PrisonersDilemma,
    PublicGoodsGame,
    StagHunt,
    UltimatumGame,
    get_game,
    run_brlhf_across_games,
    thesis_validation_summary,
)
from metrics.brlhf_standalone import (
    BRLHFMetric,
    BRLHFResult,
    BRLHFGroundingEffect,
    jsd_from_dists,
    make_prisoners_dilemma_metric,
    make_public_goods_metric,
    make_stag_hunt_metric,
    make_ultimatum_metric,
)


# ── Social Dilemma game tests ─────────────────────────────────────────────────


class TestPrisonersDilemma:
    def setup_method(self):
        self.game = PrisonersDilemma()

    def test_name(self):
        assert "Prisoner" in self.game.name

    def test_actions(self):
        assert set(self.game.actions) == {"cooperate", "defect"}

    def test_nash_is_full_defection(self):
        assert self.game.nash_equilibrium["defect"] == 1.0
        assert self.game.nash_equilibrium["cooperate"] == 0.0

    def test_social_optimum_is_full_cooperation(self):
        assert self.game.social_optimum["cooperate"] == 1.0

    def test_brlhf_all_cooperate(self):
        # RLHF prediction: cooperate 100% → max bias
        b = self.game.compute_brlhf({"cooperate": 1.0, "defect": 0.0})
        assert b == pytest.approx(0.5, abs=0.01)

    def test_brlhf_uniform_is_zero(self):
        b = self.game.compute_brlhf({"cooperate": 0.5, "defect": 0.5})
        assert b == pytest.approx(0.0, abs=0.001)

    def test_brlhf_vs_nash(self):
        # Full cooperation vs Nash (defect) = maximum distance
        b = self.game.compute_brlhf({"cooperate": 1.0, "defect": 0.0}, reference="nash")
        assert b == pytest.approx(1.0, abs=0.01)

    def test_evaluate_classifies_over_coop(self):
        result = self.game.evaluate({"cooperate": 0.90, "defect": 0.10})
        assert result.cooperation_direction == "over"
        assert result.brlhf_vs_uniform > 0

    def test_human_baseline_is_set(self):
        assert self.game.human_baseline is not None
        assert abs(sum(self.game.human_baseline.values()) - 1.0) < 0.01


class TestStagHunt:
    def setup_method(self):
        self.game = StagHunt()

    def test_actions(self):
        assert set(self.game.actions) == {"stag", "hare"}

    def test_mixed_nash_is_fifty_fifty(self):
        assert self.game.nash_equilibrium["stag"] == pytest.approx(0.5, abs=0.01)

    def test_brlhf_always_stag(self):
        b = self.game.compute_brlhf({"stag": 1.0, "hare": 0.0})
        assert b == pytest.approx(0.5, abs=0.01)

    def test_evaluate_over_coop_when_high_stag(self):
        result = self.game.evaluate({"stag": 0.95, "hare": 0.05})
        assert result.cooperation_direction == "over"


class TestPublicGoodsGame:
    def setup_method(self):
        self.game = PublicGoodsGame()

    def test_actions(self):
        assert set(self.game.actions) == {"work", "save", "cooperate"}

    def test_nash_zero_cooperation(self):
        assert self.game.nash_equilibrium["cooperate"] == 0.0

    def test_social_optimum_full_cooperation(self):
        assert self.game.social_optimum["cooperate"] == 1.0

    def test_brlhf_high_cooperation(self):
        # 96% cooperation → B_RLHF close to max
        b = self.game.compute_brlhf({"work": 0.02, "save": 0.02, "cooperate": 0.96})
        assert b > 0.5

    def test_brlhf_calibrated_cooperation(self):
        # ~50% cooperation → low B_RLHF vs uniform
        b = self.game.compute_brlhf({"work": 0.30, "save": 0.20, "cooperate": 0.50})
        assert b < 0.20


class TestUltimatumGame:
    def setup_method(self):
        self.game = UltimatumGame()

    def test_actions(self):
        assert set(self.game.actions) == {"low_offer", "fair_offer", "high_offer"}

    def test_nash_is_low_offer(self):
        assert self.game.nash_equilibrium["low_offer"] == 1.0

    def test_brlhf_all_high_offer(self):
        b = self.game.compute_brlhf({"low_offer": 0.0, "fair_offer": 0.0, "high_offer": 1.0})
        assert b > 0.4


class TestGetGame:
    def test_returns_correct_game(self):
        game = get_game("prisoners_dilemma")
        assert isinstance(game, PrisonersDilemma)

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown game"):
            get_game("nonexistent_game")


class TestRunBrlhfAcrossGames:
    def test_runs_all_four_games(self):
        dists = {
            "prisoners_dilemma": {"cooperate": 0.85, "defect": 0.15},
            "stag_hunt": {"stag": 0.90, "hare": 0.10},
            "public_goods": {"work": 0.10, "save": 0.10, "cooperate": 0.80},
            "ultimatum": {"low_offer": 0.05, "fair_offer": 0.25, "high_offer": 0.70},
        }
        results = run_brlhf_across_games(dists)
        assert len(results) == 4
        for name, result in results.items():
            assert result.brlhf_vs_uniform >= 0.0
            assert result.cooperation_direction in {"over", "under", "calibrated"}

    def test_thesis_validation_on_universal_bias(self):
        # All games show strong cooperative bias — B_RLHF > 0.05 in all
        dists = {
            "prisoners_dilemma": {"cooperate": 0.90, "defect": 0.10},
            "stag_hunt": {"stag": 0.92, "hare": 0.08},
            "public_goods": {"work": 0.05, "save": 0.05, "cooperate": 0.90},
            "ultimatum": {"low_offer": 0.03, "fair_offer": 0.17, "high_offer": 0.80},
        }
        results = run_brlhf_across_games(dists)
        thesis = thesis_validation_summary(results)
        assert thesis["n_games_tested"] == 4
        assert thesis["n_bias_confirmed_brlhf_gt_005"] == 4
        # Thesis fully supported requires all games to also show "over" direction
        # For games where the distinction is tight (ultimatum: nash=0% high_offer vs obs=80%)
        # the "over" direction should be classified correctly
        assert thesis["n_over_cooperation"] >= 3  # at minimum PD + stag + PGG


# ── Standalone B_RLHF metric tests ───────────────────────────────────────────


class TestBRLHFMetric:
    def setup_method(self):
        self.metric = make_prisoners_dilemma_metric()

    def test_uniform_gives_zero(self):
        result = self.metric.compute({"cooperate": 0.5, "defect": 0.5})
        assert result.brlhf == pytest.approx(0.0, abs=0.001)

    def test_all_cooperate_gives_max(self):
        result = self.metric.compute({"cooperate": 1.0, "defect": 0.0})
        assert result.brlhf == pytest.approx(0.5, abs=0.01)

    def test_all_defect_gives_max(self):
        result = self.metric.compute({"cooperate": 0.0, "defect": 1.0})
        assert result.brlhf == pytest.approx(0.5, abs=0.01)

    def test_over_cooperative_direction(self):
        result = self.metric.compute({"cooperate": 0.85, "defect": 0.15})
        assert result.bias_direction == "over_cooperative"

    def test_calibrated_direction(self):
        result = self.metric.compute({"cooperate": 0.50, "defect": 0.50})
        assert result.bias_direction == "calibrated"

    def test_custom_reference(self):
        human_baseline = {"cooperate": 0.47, "defect": 0.53}
        result = self.metric.compute({"cooperate": 0.90, "defect": 0.10}, human_baseline)
        assert result.reference_name == "custom"
        assert result.brlhf > 0

    def test_counts_normalized_correctly(self):
        # Raw counts should give same result as normalized probabilities
        result_counts = self.metric.compute({"cooperate": 90, "defect": 10})
        result_probs = self.metric.compute({"cooperate": 0.90, "defect": 0.10})
        assert result_counts.brlhf == pytest.approx(result_probs.brlhf, abs=0.001)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute({})

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            self.metric.compute({"cooperate": 0.0, "defect": 0.0})

    def test_grounding_effect_direction_confirmed(self):
        effect = self.metric.grounding_effect(
            condition_a={"cooperate": 0.90, "defect": 0.10},
            condition_b={"cooperate": 0.52, "defect": 0.48},
        )
        assert effect.direction_confirmed
        assert effect.absolute_reduction > 0
        assert effect.relative_reduction_pct > 0

    def test_grounding_effect_inverse(self):
        effect = self.metric.grounding_effect(
            condition_a={"cooperate": 0.52, "defect": 0.48},
            condition_b={"cooperate": 0.90, "defect": 0.10},
        )
        assert not effect.direction_confirmed
        assert effect.relative_reduction_pct < 0

    def test_audit_model_returns_full_dict(self):
        audit = self.metric.audit_model(
            model_name="TestModel",
            observed_ungrounded={"cooperate": 0.85, "defect": 0.15},
            observed_grounded={"cooperate": 0.52, "defect": 0.48},
            human_baseline={"cooperate": 0.47, "defect": 0.53},
        )
        assert "brlhf_vs_uniform" in audit
        assert "brlhf_vs_human" in audit
        assert "brlhf_reduction_pct" in audit
        assert audit["grounding_direction_confirmed"] is True


class TestBRLHFPublicGoods:
    def setup_method(self):
        self.metric = make_public_goods_metric()

    def test_bgf_pilot_condition_a(self):
        # Cond A pilot: 96.2% cooperation → high B_RLHF
        result = self.metric.compute({"work": 0.02, "save": 0.01, "cooperate": 0.97})
        assert result.brlhf > 0.50

    def test_bgf_pilot_condition_b(self):
        # Cond B pilot: 58% cooperation → lower B_RLHF
        result = self.metric.compute({"work": 0.25, "save": 0.17, "cooperate": 0.58})
        assert result.brlhf < 0.30

    def test_grounding_reduces_brlhf(self):
        effect = self.metric.grounding_effect(
            condition_a={"work": 0.02, "save": 0.01, "cooperate": 0.97},
            condition_b={"work": 0.25, "save": 0.17, "cooperate": 0.58},
        )
        assert effect.direction_confirmed
        assert effect.relative_reduction_pct > 40  # expect ~60% reduction


class TestJSD:
    def test_identical_distributions_zero(self):
        p = {"a": 0.5, "b": 0.5}
        assert jsd_from_dists(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_distributions_one(self):
        p = {"a": 1.0, "b": 0.0}
        q = {"a": 0.0, "b": 1.0}
        assert jsd_from_dists(p, q) == pytest.approx(1.0, abs=0.01)

    def test_symmetric(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.4, "b": 0.6}
        assert jsd_from_dists(p, q) == pytest.approx(jsd_from_dists(q, p), abs=1e-9)
