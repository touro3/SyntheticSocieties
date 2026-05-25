from __future__ import annotations

import logging
import random
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Candidate ESS survey-weight columns, in preference order:
#   anweight — analysis weight (combines design + post-strat + population)
#   pspwght  — post-stratification weight
#   pweight  — population weight
#   dweight  — design weight
# These are present in the raw ESS CSV but absent from the current
# `data/ess_clean.parquet` extract (which drops them during cleaning).
# When any are present, sample_empirical_rows will use them as p= for
# rng.choice; otherwise it falls back to uniform sampling and logs a
# one-time warning so the run knows weighting was not applied.
_WEIGHT_CANDIDATES = ("anweight", "pspwght", "pweight", "dweight")


def sample_age(min_age: int, max_age: int, rng: random.Random | None = None) -> int:
    """Sample an age uniformly from [min_age, max_age].

    Args:
        rng: Optional local RNG for deterministic sampling. When None,
             falls back to the global ``random`` module (seeded via
             ``set_global_seed()`` in experiment scripts).
    """
    _rng = rng or random
    return _rng.randint(min_age, max_age)


def sample_income(base_income: float, income_step: float, index: int) -> float:
    return float(base_income + income_step * index)


def sample_empirical_rows(
    parquet_path: str,
    n: int,
    mode: str = "resample",
    seed: Optional[int] = None,
    country_filter: Optional[list[str]] = None,
    exclude_countries: Optional[list[str]] = None,
    weight_column: Optional[str] = "auto",
) -> list[dict[str, Any]]:
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

    # Resolve survey-weight column.
    p_weights: Optional[np.ndarray] = None
    chosen_weight: Optional[str] = None
    if weight_column is not None:
        if weight_column == "auto":
            for cand in _WEIGHT_CANDIDATES:
                if cand in df.columns:
                    chosen_weight = cand
                    break
        elif weight_column in df.columns:
            chosen_weight = weight_column
        else:
            logger.warning(
                "sample_empirical_rows: requested weight_column=%r not present in parquet; "
                "falling back to uniform sampling.",
                weight_column,
            )

    if chosen_weight is not None:
        w = df[chosen_weight].to_numpy(dtype=float)
        # Drop NaN/negative entries by zeroing them so they cannot be selected.
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
        total = float(w.sum())
        if total > 0:
            p_weights = w / total
            logger.info("sample_empirical_rows: applying ESS survey weights from '%s'.", chosen_weight)
        else:
            logger.warning(
                "sample_empirical_rows: weight column '%s' sums to 0; falling back to uniform.",
                chosen_weight,
            )
    else:
        if weight_column == "auto":
            # One-time per-call info — the parquet is just unweighted.
            logger.info(
                "sample_empirical_rows: no survey-weight column found in %s "
                "(checked %s); using uniform sampling. ESS is a stratified survey — "
                "consider rebuilding the parquet with anweight/pspwght for unbiased marginals.",
                parquet_path,
                _WEIGHT_CANDIDATES,
            )

    if mode == "subsample" and n <= len(df) and p_weights is None:
        indices = rng.choice(len(df), size=n, replace=False)
    elif mode == "subsample" and n <= len(df):
        # Weighted subsampling — without replacement requires numpy >= 1.7.
        indices = rng.choice(len(df), size=n, replace=False, p=p_weights)
    else:
        indices = rng.choice(len(df), size=n, replace=True, p=p_weights)

    sampled = df.iloc[indices].reset_index(drop=True)
    return sampled.to_dict(orient="records")


# ── Marginal resampling for NaN substitution (audit A1.2) ────────────────────


def build_marginal_samplers(
    parquet_path: str,
    columns: list[str],
    weight_column: Optional[str] = "auto",
) -> dict[str, tuple[np.ndarray, Optional[np.ndarray]]]:
    """Return per-column (values, probs) tables for sampling-from-marginal.

    For each requested column, NaN rows are dropped and the empirical
    distribution is summarised as a value array and a matching probability
    array (weighted by the survey weight column if available, else uniform).
    Use the returned tables with ``sample_from_marginal`` to replace NaN
    cells with a draw from the column's own marginal — strictly better than
    the historical fixed-default behaviour (which spiked the median bin).

    Returns a dict ``{col: (values, probs)}``. Missing columns are skipped
    silently (caller checks ``col in result``).
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    chosen_weight: Optional[str] = None
    if weight_column == "auto":
        for cand in _WEIGHT_CANDIDATES:
            if cand in df.columns:
                chosen_weight = cand
                break
    elif weight_column is not None and weight_column in df.columns:
        chosen_weight = weight_column

    out: dict[str, tuple[np.ndarray, Optional[np.ndarray]]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        if chosen_weight is not None:
            sub = df[[col, chosen_weight]].dropna(subset=[col])
            w = sub[chosen_weight].to_numpy(dtype=float)
            w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
            total = float(w.sum())
            probs = (w / total) if total > 0 else None
        else:
            sub = df[[col]].dropna(subset=[col])
            probs = None
        values = sub[col].to_numpy()
        if len(values) == 0:
            continue
        out[col] = (values, probs)
    return out


def sample_from_marginal(
    marginal: tuple[np.ndarray, Optional[np.ndarray]],
    rng: np.random.Generator,
) -> Any:
    """Draw a single value from a (values, probs) table returned by
    ``build_marginal_samplers``. ``rng`` must be a numpy Generator for
    reproducibility (use ``np.random.default_rng(seed)``)."""
    values, probs = marginal
    idx = rng.choice(len(values), p=probs)
    return values[idx]
