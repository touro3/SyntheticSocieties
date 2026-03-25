"""Tests for empirical population generation from ESS data."""

from pathlib import Path

import pytest

from decision.mock_policy import MockPolicy


@pytest.fixture
def ess_clean_path():
    path = Path("data/ess_clean.parquet")
    if not path.exists():
        pytest.skip("ESS clean data not found. Run: python scripts/ingest_ess.py")
    return str(path)


@pytest.fixture
def empirical_config(ess_clean_path):
    return {
        "project": {"seed": 42},
        "simulation": {"population_size": 10},
        "data": {
            "ess_clean_path": ess_clean_path,
            "sample_mode": "resample",
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


def test_generate_empirical_population_size(empirical_config):
    from population.generator import generate_empirical_population
    agents = generate_empirical_population(empirical_config, MockPolicy())
    assert len(agents) == 10


def test_empirical_agents_have_ess_attributes(empirical_config):
    from population.generator import generate_empirical_population
    agents = generate_empirical_population(empirical_config, MockPolicy())
    agent = agents[0]

    # Should have ESS-derived attributes (at least some non-None)
    ess_attrs = [
        agent.profile.country,
        agent.profile.trust_people,
        agent.profile.life_satisfaction,
        agent.profile.happiness,
    ]
    non_none = [a for a in ess_attrs if a is not None]
    assert len(non_none) >= 2, "Agent should have ESS-derived attributes"


def test_empirical_population_seed_reproducibility(empirical_config):
    from population.generator import generate_empirical_population
    agents1 = generate_empirical_population(empirical_config, MockPolicy())
    agents2 = generate_empirical_population(empirical_config, MockPolicy())

    for a1, a2 in zip(agents1, agents2):
        assert a1.profile.age == a2.profile.age
        assert a1.profile.trust_people == a2.profile.trust_people


def test_empirical_agents_valid_ranges(empirical_config):
    from population.generator import generate_empirical_population
    agents = generate_empirical_population(empirical_config, MockPolicy())

    for agent in agents:
        p = agent.profile
        if p.trust_people is not None:
            assert 0 <= p.trust_people <= 1, f"trust_people out of range: {p.trust_people}"
        if p.life_satisfaction is not None:
            assert 0 <= p.life_satisfaction <= 1, f"life_satisfaction out of range: {p.life_satisfaction}"
        if p.risk_tolerance is not None:
            assert 0 <= p.risk_tolerance <= 1, f"risk_tolerance out of range: {p.risk_tolerance}"


def test_empirical_population_backward_compatible():
    """Original generate_population should still work unchanged."""
    from population.generator import generate_population
    from utils.io import set_global_seed

    config = {
        "simulation": {"population_size": 5},
        "agent_defaults": {
            "min_age": 25, "max_age": 60,
            "base_income": 1000.0, "income_step": 100.0,
            "education": "college", "occupation": "worker",
            "location": "italy", "political_preference": "center",
            "risk_tolerance": 0.5, "social_class": "middle",
            "initial_wealth": 50.0, "wealth_step": 10.0,
            "memory_size": 10,
        },
    }

    set_global_seed(42)
    agents = generate_population(config, MockPolicy())
    assert len(agents) == 5
    assert agents[0].profile.agent_id == "agent_0"
    # ESS attributes should be None (backward compat)
    assert agents[0].profile.trust_people is None
    assert agents[0].profile.country is None


def test_agent_profile_to_persona_dict(empirical_config):
    from population.generator import generate_empirical_population
    agents = generate_empirical_population(empirical_config, MockPolicy())

    persona = agents[0].profile.to_persona_dict()
    assert "id" in persona
    assert "age" in persona
    assert "income" in persona
    # Should include ESS attributes if present
    assert isinstance(persona, dict)
