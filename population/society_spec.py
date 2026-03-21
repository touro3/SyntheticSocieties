from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Band = Literal["very_low", "low", "moderate", "high", "very_high"]
AgeProfile = Literal["young", "mixed", "aging", "elderly"]
Urbanization = Literal["urban", "suburban", "rural", "mixed"]
PoliticalBand = Literal["left", "center_left", "center", "center_right", "right", "mixed"]


class SocietySpec(BaseModel):
    narrative: str = Field(..., description="Original natural-language request.")
    countries: Optional[List[str]] = None
    age_profile: Optional[AgeProfile] = None
    urbanization: Optional[Urbanization] = None
    trust_people_band: Optional[Band] = None
    trust_institutions_band: Optional[Band] = None
    social_activity_band: Optional[Band] = None
    religiosity_band: Optional[Band] = None
    political_orientation_band: Optional[PoliticalBand] = None
    risk_tolerance_band: Optional[Band] = None
    competitiveness_band: Optional[Band] = None
    target_population_size: int = 50
    notes: list[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    dataset_name: str
    selected_datasets: list[str]
    matched_rows: int
    base_rows: int
    min_cohort_size: int
    requested_filters: dict
    active_filters: dict
    variable_summaries: dict
    evidence_snippets: list[str]
    relaxation_trace: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
