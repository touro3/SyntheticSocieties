"""Shared helper functions for population generation.

Consolidates safe type conversion and ESS variable mapping functions
that were duplicated between generator.py and persona_synthesizer.py.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert to float, returning default on NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert to int, returning default on NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def clamp01(val: Optional[float]) -> Optional[float]:
    """Clamp a value to [0, 1] or return None."""
    if val is None:
        return None
    return max(0.0, min(1.0, val))


def safe_normalized_float(
    val: Any,
    scale_min: float,
    scale_max: float,
    default: Optional[float] = None,
) -> Optional[float]:
    """Convert a value from [scale_min, scale_max] to [0, 1]. Clamps result."""
    f = safe_float(val, default=None)
    if f is None:
        return default
    normalized = (f - scale_min) / (scale_max - scale_min)
    return max(0.0, min(1.0, normalized))


def safe_mean(values: list[Any]) -> Optional[float]:
    """Compute mean of non-None, non-NaN values. Returns None if all missing."""
    valid: list[float] = []
    for v in values:
        f = safe_float(v)
        if f is not None:
            valid.append(f)
    return sum(valid) / len(valid) if valid else None


# ── ESS variable mapping functions ───────────────────────────────────────────

_EDUCATION_MAP: dict[int, str] = {
    1: "less_than_lower_secondary",
    2: "lower_secondary",
    3: "upper_secondary",
    4: "post_secondary",
    5: "short_cycle_tertiary",
    6: "bachelor",
    7: "master_or_higher",
}


def map_education(level: Any, default: str = "upper_secondary") -> str:
    """Map ES-ISCED numeric level to string."""
    key = safe_int(level)
    if key is None:
        return default
    return _EDUCATION_MAP.get(key, default)


_LOCATION_MAP: dict[int, str] = {
    1: "big_city",
    2: "suburbs",
    3: "town",
    4: "village",
    5: "countryside",
}


def map_location(urbanization: Any, default: str = "town") -> str:
    """Map ESS domicile type to location string."""
    key = safe_int(urbanization)
    if key is None:
        return default
    return _LOCATION_MAP.get(key, default)


def map_political(left_right: Any, default: str = "center") -> str:
    """Map left-right scale (0-1 normalized) to preference string."""
    val = safe_float(left_right)
    if val is None:
        return default
    if val < 0.3:
        return "left"
    if val < 0.45:
        return "center-left"
    if val < 0.55:
        return "center"
    if val < 0.7:
        return "center-right"
    return "right"


def trust_institutions_mean(row: Mapping[str, Any]) -> Optional[float]:
    """Canonical institutional-trust mean used by both ESS generator paths.

    Averages the four ESS institutional-trust items present in the cleaned
    parquet (parliament / legal / police / politicians). NaN cells are
    dropped (not zero-substituted) so a single missing trust item doesn't
    pull the mean toward zero.

    Returns None only if all four items are missing for the row.

    Replaces the previous divergence where ``generator.py`` averaged 3 cols
    and ``persona_synthesizer._mean_institutions`` referenced a non-existent
    ``trust_institutions`` column, silently dropping it to a 3-col mean.
    """
    return safe_mean(
        [
            row.get("trust_parliament"),
            row.get("trust_legal"),
            row.get("trust_police"),
            row.get("trust_politicians"),
        ]
    )


def income_from_decile(
    income_decile: Any,
    base_income: float = 400.0,
    formula: str = "canonical",
) -> float:
    """Single source of truth for the ESS decile → income mapping.

    Two historical formulas existed in the codebase (audit A1.3):

    * ``"canonical"`` (default, used by ``persona_synthesizer``):
      ``income = decile * base_income``. With ``base_income = 400``, an
      agent with decile = 5 receives income 2000.
    * ``"legacy_generator"`` (used by ``generator.generate_empirical_population``
      to preserve already-published experiment numbers): ``income = decile *
      base_income * 2``. With ``base_income = 1000`` and decile = 5, this
      yields income 10000 — a 5× higher absolute scale than ``canonical``.

    Both branches share NaN handling: missing deciles fall back to the
    median bin 5.0 (callers must track NaN rates upstream — see
    ``population/generator.py`` and ``population/persona_synthesizer.py``).

    To unify the two paths in future, set ``formula="canonical"`` at every
    call site and pick a single ``base_income`` everywhere — this changes
    headline numbers and should be done as a single deliberate sweep.
    """
    val = safe_float(income_decile, default=5.0)
    # safe_float with non-None default always returns a float.
    assert val is not None
    if formula == "legacy_generator":
        return float(val) * float(base_income) * 2.0
    if formula == "canonical":
        return float(val) * float(base_income)
    raise ValueError(f"income_from_decile: unknown formula {formula!r}")


def wealth_from_decile(
    income_decile: Any,
    initial_wealth: float = 50.0,
    wealth_step: float = 10.0,
) -> float:
    """Canonical initial-wealth mapping from ESS decile.

    ``wealth = initial_wealth + (decile / 10) * wealth_step * 10``. With the
    defaults (``initial_wealth = 50``, ``wealth_step = 10``), a decile-5
    agent starts at wealth 100. NaN deciles fall back to 5.0 (median).

    Both ``generator`` and ``persona_synthesizer`` use this formula — it
    was already aligned before audit A1.3, but is centralised here so
    future readers don't have to compare two copies.
    """
    val = safe_float(income_decile, default=5.0)
    assert val is not None
    return float(initial_wealth) + (float(val) / 10.0) * float(wealth_step) * 10.0


def map_social_class(income_decile: Any, default: str = "middle") -> str:
    """Map income decile to social class string."""
    val = safe_int(income_decile)
    if val is None:
        return default
    if val <= 3:
        return "lower"
    if val <= 6:
        return "middle"
    return "upper"
