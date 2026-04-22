from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
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
from population.society_spec import SocietySpec


def normalize_record(record):
    """Clamp any out-of-range [0,1] fields in a raw persona dict.

    Handles legacy data where health_status was stored on the raw ESS
    1-5 scale instead of being normalized to [0, 1].  The ESS convention
    is 1=very good … 5=very bad, so we invert: (5 - val) / 4.
    """
    hs = record.get("health_status", None)
    if hs is not None and hs > 1.0:
        record["health_status"] = max(0.0, min(1.0, (5.0 - hs) / 4.0))
    return record


class PersonaRecord(BaseModel):
    agent_id: str
    age: int
    income: float
    education: str
    occupation: str
    location: str
    political_preference: str
    risk_tolerance: float
    social_class: str
    initial_wealth: float
    gender: int | None = None
    country: str | None = None
    education_level: int | None = None
    income_decile: int | None = None
    trust_people: float | None = None
    trust_institutions: float | None = None
    political_orientation: float | None = None
    life_satisfaction: float | None = None
    happiness: float | None = None
    immigration_attitude: float | None = None
    social_activity: float | None = None
    competitiveness: float | None = None
    leadership_preference: float | None = None
    health_status: float | None = None
    religiosity: float | None = None


def _mean_institutions(row: pd.Series) -> float | None:
    vals = [
        _safe_float(row.get("trust_institutions")),
        _safe_float(row.get("trust_parliament")),
        _safe_float(row.get("trust_legal")),
        _safe_float(row.get("trust_police")),
    ]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def synthesize_ess_personas(df: pd.DataFrame, spec: SocietySpec, n: int, seed: int = 42) -> list[PersonaRecord]:
    if df.empty:
        raise ValueError("Cannot synthesize ESS personas from an empty dataframe.")

    rng = random.Random(seed)
    records: list[PersonaRecord] = []

    for i in range(n):
        row = df.sample(n=1, replace=True, random_state=rng.randint(0, 10_000_000)).iloc[0]

        income_decile = _safe_int(row.get("income_decile"), 5)
        age = _safe_int(row.get("age"), 40) or 40
        wealth = 50.0 + (income_decile / 10.0) * 50.0

        records.append(
            PersonaRecord(
                agent_id=f"agent_{i}",
                age=age,
                income=(income_decile or 5) * 400.0,
                education=_map_education(_safe_int(row.get("education_level"))),
                occupation="worker",
                location=_map_location(_safe_int(row.get("urbanization"))),
                political_preference=_map_political(_safe_float(row.get("left_right"))),
                risk_tolerance=_safe_float(row.get("risk_taking"), 0.5) or 0.5,
                social_class=_map_social_class(income_decile),
                initial_wealth=wealth,
                gender=_safe_int(row.get("gender")),
                country=row.get("country"),
                education_level=_safe_int(row.get("education_level")),
                income_decile=income_decile,
                trust_people=_safe_float(row.get("trust_people")),
                trust_institutions=_mean_institutions(row),
                political_orientation=_safe_float(row.get("left_right")),
                life_satisfaction=_safe_float(row.get("life_satisfaction")),
                happiness=_safe_float(row.get("happiness")),
                immigration_attitude=_safe_float(row.get("immigration_same_ethnicity")),
                social_activity=_safe_float(row.get("social_meeting_freq")),
                competitiveness=_safe_float(row.get("competitiveness")),
                leadership_preference=_safe_float(row.get("leadership_preference")),
                health_status=_safe_normalized_float(row.get("self_rated_health"), 5.0, 1.0),
                religiosity=1.0
                if _safe_int(row.get("religious_belonging")) == 1
                else 0.0
                if _safe_int(row.get("religious_belonging")) == 2
                else None,
            )
        )

    return records


def _sample_band(rng: random.Random, band: str | None, fallback: tuple[float, float] = (0.35, 0.65)) -> float:
    band_ranges = {
        "very_low": (0.0, 0.15),
        "low": (0.15, 0.35),
        "moderate": (0.35, 0.65),
        "high": (0.65, 0.85),
        "very_high": (0.85, 1.0),
    }
    low, high = band_ranges.get(band or "moderate", fallback)
    return rng.uniform(low, high)


def synthesize_spec_personas(spec: SocietySpec, n: int, seed: int = 42) -> list[PersonaRecord]:
    rng = random.Random(seed)
    records: list[PersonaRecord] = []

    for i in range(n):
        if spec.age_profile == "young":
            age = rng.randint(18, 35)
        elif spec.age_profile == "aging":
            age = rng.randint(50, 72)
        elif spec.age_profile == "elderly":
            age = rng.randint(60, 85)
        else:
            age = rng.randint(25, 65)

        if spec.urbanization == "urban":
            location = rng.choice(["big_city", "suburbs"])
        elif spec.urbanization == "suburban":
            location = rng.choice(["suburbs", "town"])
        elif spec.urbanization == "rural":
            location = rng.choice(["village", "countryside"])
        else:
            location = rng.choice(["big_city", "suburbs", "town", "village"])

        pol_value_map = {
            "left": 0.20,
            "center_left": 0.40,
            "center": 0.50,
            "center_right": 0.62,
            "right": 0.82,
            "mixed": 0.50,
            None: 0.50,
        }
        pol_value = pol_value_map.get(spec.political_orientation_band, 0.50)

        income_decile = rng.randint(3, 8)
        wealth = 50.0 + income_decile * 5.0

        records.append(
            PersonaRecord(
                agent_id=f"agent_{i}",
                age=age,
                income=float(income_decile * 450.0),
                education=rng.choice(["lower_secondary", "upper_secondary", "bachelor"]),
                occupation="worker",
                location=location,
                political_preference=_map_political(pol_value),
                risk_tolerance=_sample_band(rng, spec.risk_tolerance_band),
                social_class=_map_social_class(income_decile),
                initial_wealth=wealth,
                gender=rng.choice([1, 2]),
                country=(spec.countries[0] if spec.countries else "AT"),
                education_level=rng.choice([2, 3, 4, 6]),
                income_decile=income_decile,
                trust_people=_sample_band(rng, spec.trust_people_band),
                trust_institutions=_sample_band(rng, spec.trust_institutions_band),
                political_orientation=pol_value,
                life_satisfaction=_sample_band(rng, "moderate"),
                happiness=_sample_band(rng, "moderate"),
                immigration_attitude=_sample_band(rng, "moderate"),
                social_activity=_sample_band(rng, spec.social_activity_band),
                competitiveness=_sample_band(rng, spec.competitiveness_band),
                leadership_preference=_sample_band(rng, "moderate"),
                health_status=_sample_band(rng, "moderate"),
                religiosity=_sample_band(rng, spec.religiosity_band),
            )
        )

    return records


def build_persona_natural_language(record: PersonaRecord) -> str:
    """Convert a PersonaRecord into a natural-language persona description.

    Mirrors MiroFish's OasisProfileGenerator: builds a richly worded persona
    string from the structured ESS-derived attributes so the LLM receives a
    human-readable profile rather than raw numbers.

    This description can be enriched further by ``enrich_persona_from_graph()``
    before being injected into the prompt.
    """
    parts: list[str] = []

    age_str = f"{record.age}-year-old"
    if record.gender == 1:
        gender_str = "man"
    elif record.gender == 2:
        gender_str = "woman"
    else:
        gender_str = "person"

    country_str = f" from {record.country}" if record.country else ""
    parts.append(f"{age_str} {gender_str}{country_str}.")

    if record.education:
        parts.append(f"Education: {record.education}.")

    if record.location:
        parts.append(f"Lives in a {record.location} area.")

    if record.social_class:
        parts.append(f"Social class: {record.social_class}.")

    # Trust attributes
    if record.trust_people is not None:
        level = _trust_word(record.trust_people)
        parts.append(f"Has {level} trust in other people ({record.trust_people:.2f}).")

    if record.trust_institutions is not None:
        level = _trust_word(record.trust_institutions)
        parts.append(f"Has {level} trust in institutions ({record.trust_institutions:.2f}).")

    if record.risk_tolerance is not None:
        level = _trust_word(record.risk_tolerance)
        parts.append(f"Risk tolerance is {level} ({record.risk_tolerance:.2f}).")

    if record.life_satisfaction is not None:
        level = _trust_word(record.life_satisfaction)
        parts.append(f"Life satisfaction is {level} ({record.life_satisfaction:.2f}).")

    if record.social_activity is not None:
        level = _trust_word(record.social_activity)
        parts.append(f"Social activity level is {level}.")

    return " ".join(parts)


def _trust_word(value: float) -> str:
    """Map a [0,1] score to a descriptive adjective."""
    if value < 0.2:
        return "very low"
    if value < 0.4:
        return "low"
    if value < 0.6:
        return "moderate"
    if value < 0.8:
        return "high"
    return "very high"


def enrich_persona_from_graph(
    record: PersonaRecord,
    graph_rag: object,
    base_persona: str | None = None,
) -> str:
    """Prepend graph-derived social context to a persona description.

    Mirrors MiroFish's OasisProfileGenerator enrichment step: before writing
    the final persona text, the system queries the knowledge graph for entity
    relationships and injects that relational context.  Here we query the live
    GraphRAG for the agent's social position in the cooperation network.

    If the graph is not yet initialised (e.g. round 0 of a fresh simulation)
    the function returns the base persona unchanged — the enrichment degrades
    gracefully when no graph data is available.

    Args:
        record:      The agent's PersonaRecord (ESS-derived attributes).
        graph_rag:   A ``GraphRAG`` instance from ``decision.graph_rag``.
        base_persona: Pre-built natural-language description.  If None,
                      ``build_persona_natural_language(record)`` is called.

    Returns:
        Enriched persona string: graph context block + ESS persona block.
    """
    if base_persona is None:
        base_persona = build_persona_natural_language(record)

    # Bail out gracefully if graph is uninitialised or missing the method
    if not getattr(graph_rag, "_initialized", False):
        return base_persona

    social_ctx: str = ""
    if hasattr(graph_rag, "get_social_context"):
        try:
            social_ctx = graph_rag.get_social_context(record.agent_id)
        except Exception:
            social_ctx = ""

    if not social_ctx or "not yet initialized" in social_ctx or "no recorded" in social_ctx:
        return base_persona

    # Prefix the graph-derived social context as a bracketed block so the LLM
    # can distinguish it from the static ESS persona.
    return f"[Social network context] {social_ctx}\n\n{base_persona}"


def enrich_all_personas(
    records: list[PersonaRecord],
    graph_rag: object,
) -> dict[str, str]:
    """Batch-enrich all persona records with graph context.

    Returns a dict mapping agent_id → enriched persona string.
    Agents with no graph data get the plain ESS-derived persona.
    """
    return {record.agent_id: enrich_persona_from_graph(record, graph_rag) for record in records}


def save_persona_records(records: Iterable[PersonaRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def load_persona_records(path: str | Path) -> list[PersonaRecord]:
    path = Path(path)
    records: list[PersonaRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                raw = normalize_record(raw)
                records.append(PersonaRecord(**raw))
    return records


def persona_records_to_agents(records: list[PersonaRecord], policy, memory_size: int = 10) -> list[Agent]:
    agents: list[Agent] = []

    for record in records:
        profile = AgentProfile(
            agent_id=record.agent_id,
            age=record.age,
            income=record.income,
            education=record.education,
            occupation=record.occupation,
            location=record.location,
            political_preference=record.political_preference,
            risk_tolerance=record.risk_tolerance,
            social_class=record.social_class,
            gender=record.gender,
            country=record.country,
            education_level=record.education_level,
            income_decile=record.income_decile,
            trust_people=record.trust_people,
            trust_institutions=record.trust_institutions,
            political_orientation=record.political_orientation,
            life_satisfaction=record.life_satisfaction,
            happiness=record.happiness,
            immigration_attitude=record.immigration_attitude,
            social_activity=record.social_activity,
            competitiveness=record.competitiveness,
            leadership_preference=record.leadership_preference,
            health_status=record.health_status,
            religiosity=record.religiosity,
        )
        state = AgentState(wealth=record.initial_wealth)
        memory = HierarchicalMemory(max_recent=memory_size)
        agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))

    return agents
