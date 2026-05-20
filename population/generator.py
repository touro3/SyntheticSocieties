from __future__ import annotations

import hashlib
import logging
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
    safe_normalized_float as _safe_normalized_float,
)
from population.sampling import sample_age, sample_empirical_rows, sample_income
from population.schemas import PopulationSpec

logger = logging.getLogger(__name__)


def _build_memory(defaults: dict, agent_id: str, persistent_dir: str | None) -> HierarchicalMemory:
    """Construct an agent memory, opting into the disk-persistent semantic
    store only when ``agent_defaults.memory_persistent`` is set AND a target
    directory is supplied.  Otherwise behavior is byte-identical to before.
    """
    max_recent = defaults.get("memory_size", 10)
    if persistent_dir and defaults.get("memory_persistent", False):
        from pathlib import Path

        db_path = str(Path(persistent_dir) / "memory" / f"{agent_id}.db")
        return HierarchicalMemory(
            max_recent=max_recent,
            persistent_db_path=db_path,
            embedding_model=defaults.get("embedding_model", "all-MiniLM-L6-v2"),
        )
    return HierarchicalMemory(max_recent=max_recent)


def generate_population(config: dict, policy, persistent_dir: str | None = None) -> list[Agent]:
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

        memory = _build_memory(defaults, f"agent_{i}", persistent_dir)

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
        _shuffle_traits(agents, seed=config.get("project", {}).get("seed", 42))

    return agents


def generate_empirical_population(
    config: dict,
    policy,
    data_source: Optional[str] = None,
    persistent_dir: str | None = None,
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

    # Count NaN→default substitutions so distortions to marginals are visible.
    nan_counts: dict[str, int] = {"age": 0, "income_decile": 0, "left_right": 0}

    for i, row in enumerate(rows):
        # Map ESS fields to AgentProfile, with fallbacks to config defaults
        raw_age = row.get("age")
        if raw_age is None or (isinstance(raw_age, float) and raw_age != raw_age):
            nan_counts["age"] += 1
        age = _safe_int(raw_age, default=sample_age(defaults.get("min_age", 25), defaults.get("max_age", 60)))
        raw_decile = row.get("income_decile")
        if raw_decile is None or (isinstance(raw_decile, float) and raw_decile != raw_decile):
            nan_counts["income_decile"] += 1
        income = _safe_float(raw_decile, default=0.5) * defaults.get("base_income", 1000.0) * 2

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

        # Canonical institutional-trust mean (4-item, NaN-dropped) — same
        # helper as persona_synthesizer so both paths agree on this field.
        from population._helpers import trust_institutions_mean as _trust_mean

        trust_inst = _trust_mean(row)

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
        memory = _build_memory(defaults, f"agent_{i}", persistent_dir)

        agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))

    # Warn loudly when NaN substitution distorts marginals on >5% of agents.
    for field, count in nan_counts.items():
        if count and count / max(n, 1) > 0.05:
            logger.warning(
                "generate_empirical_population: %d/%d (%.1f%%) agents had NaN '%s' "
                "and were substituted with a default — empirical marginal for this "
                "field is distorted.",
                count,
                n,
                100.0 * count / n,
                field,
            )

    # ── Condition C: Counterfactual Identity ("Soul Swap") ────────────────
    if defaults.get("shuffle_traits", False):
        _shuffle_traits(agents, seed=seed)

    return agents


def generate_placebo_population(
    config: dict,
    policy,
    data_source: Optional[str] = None,
) -> list[Agent]:
    """Generate a placebo population — the semantic-isolation control.

    Demographics are drawn from real ESS respondents (so prompt heterogeneity
    matches the grounded arm exactly), but the sociological trait vector is
    independently permuted across the population: marginals preserved, joint
    structure destroyed. See :mod:`population.placebo_demographics` for the
    full research rationale.

    Mirrors :func:`generate_empirical_population` for config/seed handling so
    the three arms (empirical / placebo / synthetic) differ only in grounding.

    Args:
        config: Simulation config dict.
        policy: Decision policy to assign to agents.
        data_source: Path to cleaned ESS Parquet. If None, reads from
                     ``config["data"]["ess_clean_path"]``.
    """
    import pandas as _pd

    from population.persona_synthesizer import persona_records_to_agents
    from population.placebo_demographics import synthesize_placebo_personas

    simulation_cfg = config["simulation"]
    defaults = config["agent_defaults"]
    data_cfg = config.get("data", {})

    n = simulation_cfg["population_size"]
    seed = config.get("project", {}).get("seed", 42)

    if data_source is None:
        data_source = data_cfg.get("ess_clean_path", "data/ess_clean.parquet")

    df = _pd.read_parquet(data_source)
    records = synthesize_placebo_personas(df, n=n, seed=seed)
    agents = persona_records_to_agents(
        records,
        policy=policy,
        memory_size=defaults.get("memory_size", 10),
    )
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


def _shuffle_traits(agents: list[Agent], seed: int = 42) -> None:
    """Permute psychological trait columns independently across the population.

    Demographics (age, income, education, gender, country) remain intact.
    Each trait is shuffled independently so that the joint distribution of
    traits is also destroyed — this is the strongest "soul swap" condition.

    Determinism: a local ``random.Random`` is seeded per trait field with a
    stable SHA-256-derived sub-seed (NOT Python's salted ``hash()``), so the
    Condition C permutation is fully reproducible for a given experiment seed
    and immune to any intervening global-random consumption.
    """
    n = len(agents)
    if n < 2:
        return

    for field in _TRAIT_FIELDS:
        # Independent, stable per-field sub-seed.
        field_seed = int(hashlib.sha256(f"{seed}:{field}".encode()).hexdigest()[:8], 16)
        field_rng = random.Random(field_seed)
        # Collect current values
        values = [getattr(a.profile, field, None) for a in agents]
        field_rng.shuffle(values)
        # Re-assign
        for agent, val in zip(agents, values):
            if hasattr(agent.profile, field):
                object.__setattr__(agent.profile, field, val)
