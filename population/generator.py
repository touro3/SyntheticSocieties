from __future__ import annotations

import math
from typing import Optional

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState

from population.sampling import sample_age, sample_income, sample_empirical_rows
from population.schemas import PopulationSpec


def generate_population(config: dict, policy) -> list[Agent]:
    """Generate a synthetic population from config defaults (v0.1 — backward compatible)."""
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

        memory = HierarchicalMemory(max_recent=defaults["memory_size"])


        agents.append(
            Agent(
                profile=profile,
                state=state,
                memory=memory,
                policy=policy,
            )
        )

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
        age = _safe_int(row.get("age"), default=sample_age(
            defaults.get("min_age", 25), defaults.get("max_age", 60)
        ))
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
            religiosity=1.0 if row.get("religious_belonging") == 1 else 0.0 if row.get("religious_belonging") == 2 else None,
        )

        # Initial wealth based on income decile
        wealth = defaults.get("initial_wealth", 50.0) + (
            _safe_float(row.get("income_decile"), 5) / 10.0
        ) * defaults.get("wealth_step", 10.0) * 10

        state = AgentState(wealth=wealth)
        memory = HierarchicalMemory(max_recent=defaults.get("memory_size", 10))


        agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))

    return agents


# ── Helper functions ─────────────────────────────────────────────────────────

def _safe_float(val, default: float = None) -> Optional[float]:
    """Safely convert to float, returning default on NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = None) -> Optional[int]:
    """Safely convert to int, returning default on NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def _clamp01(val: Optional[float]) -> Optional[float]:
    """Clamp a value to [0, 1] or return None."""
    if val is None:
        return None
    return max(0.0, min(1.0, val))


def _safe_normalized_float(val, scale_min: float, scale_max: float, default: float = None) -> Optional[float]:
    """Convert a value from [scale_min, scale_max] to [0, 1]. Clamps result."""
    f = _safe_float(val, default=None)
    if f is None:
        return default
    normalized = (f - scale_min) / (scale_max - scale_min)
    return max(0.0, min(1.0, normalized))


def _safe_mean(values: list) -> Optional[float]:
    """Compute mean of non-None, non-NaN values. Returns None if all missing."""
    valid = []
    for v in values:
        f = _safe_float(v)
        if f is not None:
            valid.append(f)
    return sum(valid) / len(valid) if valid else None


def _map_education(level, default: str) -> str:
    """Map ES-ISCED numeric level to string."""
    mapping = {
        1: "less_than_lower_secondary",
        2: "lower_secondary",
        3: "upper_secondary",
        4: "post_secondary",
        5: "short_cycle_tertiary",
        6: "bachelor",
        7: "master_or_higher",
    }
    return mapping.get(_safe_int(level), default)


def _map_location(urbanization, default: str) -> str:
    """Map ESS domicile type to location string."""
    mapping = {
        1: "big_city",
        2: "suburbs",
        3: "town",
        4: "village",
        5: "countryside",
    }
    return mapping.get(_safe_int(urbanization), default)


def _map_political(left_right, default: str) -> str:
    """Map left-right scale (0-1 normalized) to preference string."""
    val = _safe_float(left_right)
    if val is None:
        return default
    if val < 0.3:
        return "left"
    if val < 0.45:
        return "center-left"
    if val < 0.55:
        return "center"
    if val < 0.7:
        return "center-right"
    return "right"


def _map_social_class(income_decile, default: str) -> str:
    """Map income decile to social class string."""
    val = _safe_int(income_decile)
    if val is None:
        return default
    if val <= 3:
        return "lower"
    if val <= 6:
        return "middle"
    return "upper"