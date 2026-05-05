from __future__ import annotations

import random as _random
import random
from typing import Optional

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from population._helpers import (
    clamp01 as _clamp01,
)
from population._helpers import (
    map_education as _map_education,
)
from population._helpers import (
    map_location as _map_location,
)
from population._helpers import (
    map_political as _map_political,
)
from population._helpers import (
    map_social_class as _map_social_class,
)
from population._helpers import (
    safe_float as _safe_float,
)
from population._helpers import (
    safe_int as _safe_int,
)
from population._helpers import (
    safe_mean as _safe_mean,
)
from population._helpers import (
    safe_normalized_float as _safe_normalized_float,
)
from population.sampling import sample_age, sample_empirical_rows, sample_income
from population.schemas import PopulationSpec


def generate_population(config: dict, policy) -> list[Agent]:
    """Generate a synthetic population from config defaults (v0.1 — backward compatible).

    When ``agent_defaults.shuffle_traits`` is True, psychological traits
    (risk_tolerance, competitiveness, trust_people, etc.) are randomly
    permuted across agents **after** generation.  This creates "Condition C"
    — the counterfactual identity condition where demographics remain intact
    but trait assignments are decorrelated.
    """
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

        state = AgentState(wealth=spec.initial_wealth + i * spec.wealth_step)

        memory = HierarchicalMemory(max_recent=defaults["memory_size"])

        agents.append(
            Agent(
                profile=profile,
                state=state,
                memory=memory,
                policy=policy,
            )
        )

    # ── Condition C: Counterfactual Identity ("Soul Swap") ────────────────
    if defaults.get("shuffle_traits", False):
        _shuffle_traits(agents)

    return agents


def generate_empirical_population(
    config: dict,
    policy,
    data_source: Optional[str] = None,
) -> list[Agent]:
    """
    Generate a population grounded in ESS empirical data.

    Each agent is constructed from a real ESS respondent row, mapping
    survey variables to AgentProfile attributes. This preserves joint
    distributions across demographic, attitudinal, and behavioral
    variables.

    Args:
        config: Simulation config dict.
        policy: Decision policy to assign to agents.
        data_source: Path to cleaned ESS Parquet. If None, reads from
                     config["data"]["ess_clean_path"].
    """
    simulation_cfg = config["simulation"]
    defaults = config["agent_defaults"]
    data_cfg = config.get("data", {})

    n = simulation_cfg["population_size"]
    seed = config.get("project", {}).get("seed", 42)

    if data_source is None:
        data_source = data_cfg.get("ess_clean_path", "data/ess_clean.parquet")

    sample_mode = data_cfg.get("sample_mode", "resample")

    rows = sample_empirical_rows(
        parquet_path=data_source,
        n=n,
        mode=sample_mode,
        seed=seed,
    )

    agents: list[Agent] = []

    for i, row in enumerate(rows):
        # Map ESS fields to AgentProfile, with fallbacks to config defaults
        age = _safe_int(row.get("age"), default=sample_age(defaults.get("min_age", 25), defaults.get("max_age", 60)))
        income = _safe_float(row.get("income_decile"), default=0.5) * defaults.get("base_income", 1000.0) * 2

        # Map education level to string
        education = _map_education(row.get("education_level"), defaults.get("education", "unknown"))

        # Map urbanization to location string
        location = _map_location(row.get("urbanization"), defaults.get("location", "unknown"))

        # Political orientation → preference string
        political_pref = _map_political(row.get("left_right"), defaults.get("political_preference", "center"))

        # Risk tolerance from ESS risk variable
        risk = _safe_float(row.get("risk_taking"), default=defaults.get("risk_tolerance", 0.5))

        # Social class from income decile
        social_class = _map_social_class(row.get("income_decile"), defaults.get("social_class", "middle"))

        # Compute trust_institutions as average of available institutional trust
        trust_inst_vars = [
            row.get("trust_parliament"),
            row.get("trust_legal"),
            row.get("trust_police"),
        ]
        trust_inst = _safe_mean(trust_inst_vars)

        profile = AgentProfile(
            agent_id=f"agent_{i}",
            age=age,
            income=income,
            education=education,
            occupation=defaults.get("occupation", "unknown"),
            location=location,
            political_preference=political_pref,
            risk_tolerance=risk,
            social_class=social_class,
            # ESS-derived attributes
            gender=_safe_int(row.get("gender")),
            country=row.get("country"),
            education_level=_safe_int(row.get("education_level")),
            income_decile=_safe_int(row.get("income_decile")),
            trust_people=_clamp01(_safe_float(row.get("trust_people"))),
            trust_institutions=_clamp01(trust_inst),
            political_orientation=_clamp01(_safe_float(row.get("left_right"))),
            life_satisfaction=_clamp01(_safe_float(row.get("life_satisfaction"))),
            happiness=_clamp01(_safe_float(row.get("happiness"))),
            immigration_attitude=_clamp01(_safe_float(row.get("immigration_same_ethnicity"))),
            social_activity=_clamp01(_safe_float(row.get("social_meeting_freq"))),
            competitiveness=_clamp01(_safe_float(row.get("competitiveness"))),
            leadership_preference=_clamp01(_safe_float(row.get("leadership_preference"))),
            health_status=_safe_normalized_float(row.get("self_rated_health"), 5.0, 1.0),
            religiosity=1.0
            if row.get("religious_belonging") == 1
            else 0.0
            if row.get("religious_belonging") == 2
            else None,
        )

        # Initial wealth based on income decile
        wealth = (
            defaults.get("initial_wealth", 50.0)
            + (_safe_float(row.get("income_decile"), 5) / 10.0) * defaults.get("wealth_step", 10.0) * 10
        )

        state = AgentState(wealth=wealth)
        memory = HierarchicalMemory(max_recent=defaults.get("memory_size", 10))

        agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))

    # ── Condition C: Counterfactual Identity ("Soul Swap") ────────────────
    if defaults.get("shuffle_traits", False):
        _shuffle_traits(agents)

    return agents


def from_seed_document(
    doc,
    n_agents: int,
    policy=None,
    backend=None,
    rng: random.Random | None = None,
    memory_size: int = 10,
):
    """Generate PersonaRecords, or Agents when a policy is supplied, from text.

    This is intentionally lightweight: ``SeedExtractor`` creates
    PersonaRecord-compatible rows and the existing persona conversion helper
    performs the normal AgentProfile/AgentState construction.
    """
    from population.persona_synthesizer import persona_records_to_agents
    from population.seed_extractor import SeedExtractor

    extractor = SeedExtractor(backend=backend)
    entities = extractor.extract(doc)
    records = extractor.to_persona_records(entities, n_agents=n_agents, rng=rng)
    if policy is None:
        return records
    return persona_records_to_agents(records, policy=policy, memory_size=memory_size)


# ── Counterfactual Identity helper ───────────────────────────────────────────

# Psychological traits that get permuted independently in Condition C.
_TRAIT_FIELDS = ("risk_tolerance", "competitiveness", "trust_people", "social_activity")


def _shuffle_traits(agents: list[Agent]) -> None:
    """Permute psychological trait columns independently across the population.

    Demographics (age, income, education, gender, country) remain intact.
    Each trait is shuffled independently so that the joint distribution of
    traits is also destroyed — this is the strongest "soul swap" condition.
    """
    n = len(agents)
    if n < 2:
        return

    for field in _TRAIT_FIELDS:
        # Collect current values
        values = [getattr(a.profile, field, None) for a in agents]
        _random.shuffle(values)
        # Re-assign
        for agent, val in zip(agents, values):
            if hasattr(agent.profile, field):
                object.__setattr__(agent.profile, field, val)
