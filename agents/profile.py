from dataclasses import dataclass
from typing import Optional

_NORMALIZED_FIELDS = (
    "trust_people", "trust_institutions", "political_orientation",
    "life_satisfaction", "happiness", "immigration_attitude",
    "social_activity", "competitiveness", "leadership_preference",
    "health_status", "religiosity",
)


@dataclass
class AgentProfile:
    """
    Agent persona attributes.

    Core attributes (v0.1 — backward compatible):
        agent_id, age, income, education, occupation, location,
        political_preference, risk_tolerance, social_class

    ESS-derived attributes (v0.2 — empirical grounding):
        gender, country, trust_level, political_orientation,
        life_satisfaction, social_activity, health_status,
        religiosity, immigration_attitude, happiness,
        competitiveness, leadership_preference, education_level
    """
    # ── Core attributes (v0.1) ───────────────────────────────────────────
    agent_id: str
    age: int
    income: float
    education: str
    occupation: str
    location: str
    political_preference: str
    risk_tolerance: float
    social_class: str

    # ── ESS-derived attributes (v0.2) ────────────────────────────────────
    # All default to None for backward compatibility with existing tests.
    gender: Optional[int] = None                  # 1=Male, 2=Female
    country: Optional[str] = None                 # ISO 3166-1 alpha-2
    education_level: Optional[int] = None         # ES-ISCED (1-7)
    income_decile: Optional[int] = None           # 1-10

    # Trust (0-1 normalized)
    trust_people: Optional[float] = None
    trust_institutions: Optional[float] = None    # average of parliament/legal/police

    # Values & attitudes (0-1 normalized)
    political_orientation: Optional[float] = None  # 0=left, 1=right
    life_satisfaction: Optional[float] = None      # 0-1
    happiness: Optional[float] = None              # 0-1
    immigration_attitude: Optional[float] = None   # 0=restrictive, 1=open

    # Social & behavioral (0-1 normalized)
    social_activity: Optional[float] = None        # 0-1
    competitiveness: Optional[float] = None        # 0-1
    leadership_preference: Optional[float] = None  # 0-1

    # Health & wellbeing
    health_status: Optional[float] = None          # 0=poor, 1=excellent
    religiosity: Optional[float] = None            # 0-1

    def __post_init__(self) -> None:
        # Validate gender
        if self.gender is not None and self.gender not in (1, 2):
            raise ValueError(f"gender must be 1 (Male) or 2 (Female), got {self.gender}")
        # Validate education level (ES-ISCED 1-7)
        if self.education_level is not None and not (1 <= self.education_level <= 7):
            raise ValueError(f"education_level must be 1-7, got {self.education_level}")
        # Validate income_decile (1-10)
        if self.income_decile is not None and not (1 <= self.income_decile <= 10):
            raise ValueError(f"income_decile must be 1-10, got {self.income_decile}")
        # Validate all [0,1] normalized fields
        for attr in _NORMALIZED_FIELDS:
            val = getattr(self, attr, None)
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0.0, 1.0], got {val}")

    def to_persona_dict(self) -> dict:
        """Return a dict describing the agent persona for LLM prompt building."""
        persona = {
            "id": self.agent_id,
            "age": self.age,
            "income": self.income,
            "education": self.education,
            "occupation": self.occupation,
            "location": self.location,
            "political_preference": self.political_preference,
            "risk_tolerance": self.risk_tolerance,
            "social_class": self.social_class,
        }
        # Add ESS attributes if available
        for attr in [
            "gender", "country", "trust_people", "trust_institutions",
            "political_orientation", "life_satisfaction", "happiness",
            "immigration_attitude", "social_activity", "competitiveness",
            "leadership_preference", "health_status", "religiosity",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                persona[attr] = val
        return persona