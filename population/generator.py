from __future__ import annotations

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from population.sampling import sample_age, sample_income
from population.schemas import PopulationSpec


def generate_population(config: dict) -> list[Agent]:
    simulation_cfg = config["simulation"]
    defaults = config["agent_defaults"]

    spec = PopulationSpec(
        population_size=simulation_cfg["population_size"],
        min_age=defaults["min_age"],
        max_age=defaults["max_age"],
        base_income=defaults["base_income"],
        income_step=defaults["income_step"],
        initial_wealth=defaults["initial_wealth"],
        wealth_step=defaults["wealth_step"],
    )

    agents: list[Agent] = []

    for i in range(spec.population_size):
        profile = AgentProfile(
            agent_id=f"agent_{i}",
            age=sample_age(spec.min_age, spec.max_age),
            income=sample_income(spec.base_income, spec.income_step, i),
            education=defaults["education"],
            occupation=defaults["occupation"],
            location=defaults["location"],
            political_preference=defaults["political_preference"],
            risk_tolerance=defaults["risk_tolerance"],
            social_class=defaults["social_class"],
        )

        state = AgentState(
            wealth=spec.initial_wealth + i * spec.wealth_step
        )

        memory = MemoryBuffer(max_items=defaults["memory_size"])
        policy = MockPolicy()

        agents.append(
            Agent(
                profile=profile,
                state=state,
                memory=memory,
                policy=policy,
            )
        )

    return agents