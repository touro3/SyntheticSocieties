from __future__ import annotations

import random
from typing import Optional

import numpy as np


def sample_age(min_age: int, max_age: int) -> int:
    return random.randint(min_age, max_age)


def sample_income(base_income: float, income_step: float, index: int) -> float:
    return float(base_income + income_step * index)


def sample_empirical_rows(
    parquet_path: str,
    n: int,
    mode: str = "resample",
    seed: Optional[int] = None,
):
    """
    Sample rows from a cleaned ESS Parquet file.

    Args:
        parquet_path: Path to the cleaned ESS Parquet file.
        n: Number of rows to sample.
        mode: "resample" (with replacement) or "subsample" (without).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, one per sampled row.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)

    if mode == "subsample" and n <= len(df):
        indices = rng.choice(len(df), size=n, replace=False)
    else:
        indices = rng.choice(len(df), size=n, replace=True)

    sampled = df.iloc[indices].reset_index(drop=True)
    return sampled.to_dict(orient="records")
