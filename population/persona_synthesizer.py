from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from population.society_spec import SocietySpec


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

from population._helpers import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    map_education as _map_education,
    map_location as _map_location,
    map_political as _map_political,
    map_social_class as _map_social_class,
    safe_mean,
)


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
                health_status=_safe_float(row.get("self_rated_health")),
                religiosity=1.0 if _safe_int(row.get("religious_belonging")) == 1 else 0.0 if _safe_int(row.get("religious_belonging")) == 2 else None,
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
                records.append(PersonaRecord(**json.loads(line)))
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
