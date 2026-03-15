from dataclasses import dataclass
from typing import Optional


@dataclass
class PopulationSpec:
    population_size: int
    min_age: int
    max_age: int
    base_income: float
    income_step: float
    initial_wealth: float
    wealth_step: float
    data_source: Optional[str] = None       # Path to ESS Parquet file
    sample_mode: str = "resample"           # "resample" (with replacement) or "subsample"