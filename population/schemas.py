from dataclasses import dataclass


@dataclass
class PopulationSpec:
    population_size: int
    min_age: int
    max_age: int
    base_income: float
    income_step: float
    initial_wealth: float
    wealth_step: float