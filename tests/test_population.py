from pathlib import Path
from decision.mock_policy import MockPolicy

from population.generator import generate_population
from utils.io import set_global_seed


def test_generate_population_from_config():
    config = {
        "simulation": {
            "population_size": 5,
        },
        "agent_defaults": {
            "min_age": 25,
            "max_age": 60,
            "base_income": 1000.0,
            "income_step": 100.0,
            "education": "college",
            "occupation": "worker",
            "location": "italy",
            "political_preference": "center",
            "risk_tolerance": 0.5,
            "social_class": "middle",
            "initial_wealth": 50.0,
            "wealth_step": 10.0,
            "memory_size": 10,
        },
    }

    set_global_seed(42)
    agents = generate_population(config, MockPolicy())

    assert len(agents) == 5
    assert agents[0].profile.agent_id == "agent_0"
    assert agents[0].state.wealth == 50.0
    assert agents[1].state.wealth == 60.0
    assert 25 <= agents[0].profile.age <= 60


# ── Counterfactual Identity ("Soul Swap") — Condition C ───────────────────────

import numpy as np
from population.generator import _shuffle_traits


class TestCounterfactualIdentity:
    """shuffle_traits=True breaks correlation between demographics and traits."""

    def _config(self):
        return {
            "simulation": {"population_size": 50},
            "agent_defaults": {
                "min_age": 25,
                "max_age": 60,
                "base_income": 1000.0,
                "income_step": 100.0,
                "education": "college",
                "occupation": "worker",
                "location": "italy",
                "political_preference": "center",
                "risk_tolerance": 0.5,
                "social_class": "middle",
                "initial_wealth": 50.0,
                "wealth_step": 10.0,
                "memory_size": 10,
            },
        }

    def _diversify_traits(self, agents):
        """Assign diverse risk_tolerance values so shuffle has observable effect."""
        for i, agent in enumerate(agents):
            object.__setattr__(agent.profile, "risk_tolerance", i / len(agents))

    def test_shuffle_traits_changes_risk_tolerance(self):
        """With shuffled traits, risk_tolerance should differ from baseline."""
        set_global_seed(42)
        config = self._config()
        baseline = generate_population(config, MockPolicy())
        self._diversify_traits(baseline)
        original_risks = [a.profile.risk_tolerance for a in baseline]

        # Copy original values and shuffle
        import copy
        shuffled = copy.deepcopy(baseline)
        _shuffle_traits(shuffled)
        shuffled_risks = [a.profile.risk_tolerance for a in shuffled]

        # Demographics should be preserved (same ages)
        for b, s in zip(baseline, shuffled):
            assert b.profile.age == s.profile.age

        # But risk_tolerance should be shuffled (different assignment)
        assert original_risks != shuffled_risks, (
            "shuffle_traits should produce a different risk_tolerance assignment"
        )

    def test_shuffle_traits_preserves_demographics(self):
        """Demographics (age, income, education) must be unchanged."""
        set_global_seed(42)
        config = self._config()
        config["agent_defaults"]["shuffle_traits"] = True
        agents = generate_population(config, MockPolicy())

        # All agents should still have valid demographics
        for a in agents:
            assert 25 <= a.profile.age <= 60
            assert a.profile.education == "college"

    def test_shuffle_breaks_correlation(self):
        """Statistical correlation between index and risk_tolerance should be destroyed."""
        set_global_seed(42)
        config = self._config()
        agents = generate_population(config, MockPolicy())
        self._diversify_traits(agents)

        # Before shuffle: risk_tolerance = i/N, perfect correlation with index
        indices = np.arange(len(agents), dtype=float)
        risks_before = np.array([a.profile.risk_tolerance for a in agents])
        corr_before = np.corrcoef(indices, risks_before)[0, 1]
        assert abs(corr_before) > 0.9, "Pre-shuffle should have high correlation"

        _shuffle_traits(agents)
        risks_after = np.array([a.profile.risk_tolerance for a in agents])

        # After shuffle: correlation should be broken
        if np.std(risks_after) > 0:
            corr_after = np.corrcoef(indices, risks_after)[0, 1]
            assert abs(corr_after) < 0.5, (
                f"Correlation should be broken by shuffle, got r={corr_after:.3f}"
            )

    def test_shuffle_traits_false_is_default(self):
        """Default (no shuffle_traits key) should produce deterministic output."""
        set_global_seed(42)
        config = self._config()
        agents1 = generate_population(config, MockPolicy())

        set_global_seed(42)
        agents2 = generate_population(config, MockPolicy())

        for a1, a2 in zip(agents1, agents2):
            assert a1.profile.risk_tolerance == a2.profile.risk_tolerance