from __future__ import annotations

import random


def sample_age(min_age: int, max_age: int) -> int:
    return random.randint(min_age, max_age)


def sample_income(base_income: float, income_step: float, index: int) -> float:
    return float(base_income + income_step * index)