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