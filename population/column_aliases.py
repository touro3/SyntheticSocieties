"""ESS column-name aliasing.

The European Social Survey distributes microdata with short, cryptic column
codes (``agea`` for age, ``gndr`` for gender, ``cntry`` for country, …). The
BGF generator and grounder consume a canonical schema (``age``, ``gender``,
``country``, …) that researchers also typically use after data cleaning.

When a user uploads ESS-native data through the API, we lowercase column
names, strip whitespace, and apply :data:`COLUMN_ALIASES` so the canonical
schema "just works" without manual renaming. The renaming is reported back
in the upload-analysis sidecar so the UI can show which inputs were
auto-mapped.

Only columns actively read by ``population/generator.py``,
``population/ess_grounding.py``, and ``population/sampling.py`` need entries
here; extending the map is a safe additive change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# ESS short code → canonical name. Keys are lowercased; the normalizer
# applies str.lower() + str.strip() before lookup, so case variants in the
# upload (``AGE``, ``Age``) collapse to ``age`` automatically without
# needing an explicit entry.
COLUMN_ALIASES: dict[str, str] = {
    # Demographics
    "agea": "age",
    "yrbrn": "year_born",
    "gndr": "gender",
    "cntry": "country",
    # Education
    "eduyrs": "education_years",
    "edulvla": "education_level",
    "edlvb": "education_level",
    "eisced": "education_level",
    "isced": "education_level",
    # Income / wealth proxies
    "hinctnta": "income_decile",
    "hinctnt": "income_decile",
    # Institutional trust battery
    "trstprl": "trust_parliament",
    "trstlgl": "trust_legal",
    "trstplc": "trust_police",
    "trstplt": "trust_politicians",
    # Interpersonal trust
    "ppltrst": "trust_people",
    # Subjective wellbeing
    "stflife": "life_satisfaction",
    "happy": "happiness",
    "health": "self_rated_health",
    # Political orientation
    "lrscale": "left_right",
    # Social participation
    "sclmeet": "social_meeting_freq",
    "rlgblg": "religious_belonging",
    # Migration attitudes
    "imsmetn": "immigration_same_ethnicity",
    "imdfetn": "immigration_diff_ethnicity",
    # Living context
    "domicil": "urbanization",
    # Survey weights — preserved so sampling.py can pick them up
    "anweight": "anweight",
    "pspwght": "pspwght",
    "pweight": "pweight",
    "dweight": "dweight",
}


def normalize_columns(
    columns: Iterable[str],
    extra_aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build a ``rename`` map from raw → canonical column names.

    Args:
        columns: The raw column names from the uploaded dataframe.
        extra_aliases: Optional additional aliases that take precedence over
            :data:`COLUMN_ALIASES`. Useful for per-survey overrides
            (e.g. SOEP, WVS) without mutating the module-level constant.

    Returns:
        A ``(rename_map, alias_hits)`` pair.

        - ``rename_map``: pass to ``DataFrame.rename(columns=...)`` to apply
          the normalization.
        - ``alias_hits``: a ``{raw → canonical}`` dict containing only the
          columns that actually matched a non-trivial alias (i.e. the rename
          changed the name). Surface this in the upload sidecar so the user
          can see which ESS codes were auto-mapped.
    """
    aliases: dict[str, str] = dict(COLUMN_ALIASES)
    if extra_aliases:
        for k, v in extra_aliases.items():
            aliases[str(k).strip().lower()] = str(v)

    rename_map: dict[str, str] = {}
    alias_hits: dict[str, str] = {}
    seen_canonical: set[str] = set()

    for original in columns:
        if original is None:
            continue
        raw = str(original)
        key = raw.strip().lower()
        canonical = aliases.get(key, key)
        # If two raw columns both map to the same canonical name (e.g. both
        # `edulvla` and `eisced` → `education_level`), keep the first one
        # and leave the rest under their raw name to avoid a collision.
        if canonical in seen_canonical:
            canonical = key
        seen_canonical.add(canonical)
        if canonical != raw:
            rename_map[raw] = canonical
        if canonical != key:
            # Pure case normalization (`AGE` → `age`) is uninteresting;
            # only report true aliasing (`agea` → `age`).
            alias_hits[raw] = canonical

    return rename_map, alias_hits


__all__ = ["COLUMN_ALIASES", "normalize_columns"]
