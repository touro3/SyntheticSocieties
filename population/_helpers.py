"""Shared helper functions for population generation.

Consolidates safe type conversion and ESS variable mapping functions
that were duplicated between generator.py and persona_synthesizer.py.
"""

from __future__ import annotations

import math
from typing import Optional


def safe_float(val, default: float = None) -> Optional[float]:
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


def safe_int(val, default: int = None) -> Optional[int]:
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
    val, scale_min: float, scale_max: float, default: float = None
) -> Optional[float]:
    """Convert a value from [scale_min, scale_max] to [0, 1]. Clamps result."""
    f = safe_float(val, default=None)
    if f is None:
        return default
    normalized = (f - scale_min) / (scale_max - scale_min)
    return max(0.0, min(1.0, normalized))


def safe_mean(values: list) -> Optional[float]:
    """Compute mean of non-None, non-NaN values. Returns None if all missing."""
    valid = []
    for v in values:
        f = safe_float(v)
        if f is not None:
            valid.append(f)
    return sum(valid) / len(valid) if valid else None


# ── ESS variable mapping functions ───────────────────────────────────────────

_EDUCATION_MAP = {
    1: "less_than_lower_secondary",
    2: "lower_secondary",
    3: "upper_secondary",
    4: "post_secondary",
    5: "short_cycle_tertiary",
    6: "bachelor",
    7: "master_or_higher",
}


def map_education(level, default: str = "upper_secondary") -> str:
    """Map ES-ISCED numeric level to string."""
    return _EDUCATION_MAP.get(safe_int(level), default)


_LOCATION_MAP = {
    1: "big_city",
    2: "suburbs",
    3: "town",
    4: "village",
    5: "countryside",
}


def map_location(urbanization, default: str = "town") -> str:
    """Map ESS domicile type to location string."""
    return _LOCATION_MAP.get(safe_int(urbanization), default)


def map_political(left_right, default: str = "center") -> str:
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


def map_social_class(income_decile, default: str = "middle") -> str:
    """Map income decile to social class string."""
    val = safe_int(income_decile)
    if val is None:
        return default
    if val <= 3:
        return "lower"
    if val <= 6:
        return "middle"
    return "upper"
