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
    country_filter: Optional[list[str]] = None,
    exclude_countries: Optional[list[str]] = None,
):
    """
    Sample rows from a cleaned ESS Parquet file.

    Args:
        parquet_path: Path to the cleaned ESS Parquet file.
        n: Number of rows to sample.
        mode: "resample" (with replacement) or "subsample" (without).
        seed: Random seed for reproducibility.
        country_filter: If given, keep only rows whose ``country`` is in this
            list (OOD train/eval splitting). ``None`` = no filtering.
        exclude_countries: If given, drop rows whose ``country`` is in this
            list. Applied after ``country_filter``. ``None`` = no exclusion.

    Returns:
        List of dicts, one per sampled row.

    Raises:
        ValueError: If a country filter eliminates every row. The local ESS
            parquet currently contains only AT microdata, so a filter to a
            non-AT country legitimately empties the frame — this is surfaced
            loudly rather than silently resampling the wrong cohort.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)

    if country_filter is not None:
        df = df[df["country"].isin(country_filter)]
    if exclude_countries is not None:
        df = df[~df["country"].isin(exclude_countries)]

    if len(df) == 0:
        raise ValueError(
            "Country filter eliminated all rows "
            f"(country_filter={country_filter}, exclude_countries={exclude_countries}). "
            "Note: the local parquet contains only AT microdata; cross-country "
            "OOD splits operate at the cluster-benchmark level (see "
            "population.ood_split), not on microdata."
        )

    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if mode == "subsample" and n <= len(df):
        indices = rng.choice(len(df), size=n, replace=False)
    else:
        indices = rng.choice(len(df), size=n, replace=True)

    sampled = df.iloc[indices].reset_index(drop=True)
    return sampled.to_dict(orient="records")
