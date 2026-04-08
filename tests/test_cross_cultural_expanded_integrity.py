from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_cross_cultural_expanded import _assert_no_wvs_columns, _parse_seeds_arg


def test_parse_seeds_arg_explicit_list():
    assert _parse_seeds_arg("7,8,9", n_seeds_fallback=20) == [7, 8, 9]


def test_parse_seeds_arg_fallback_count():
    seeds = _parse_seeds_arg(None, n_seeds_fallback=3)
    assert seeds == [42, 43, 44]


def test_assert_no_wvs_columns_ok():
    df = pd.DataFrame({"trust_people": [0.1, 0.5, 0.9], "age": [20, 35, 60]})
    _assert_no_wvs_columns(df)


def test_assert_no_wvs_columns_raises_on_leakage_column():
    df = pd.DataFrame({"trust_people": [0.2, 0.6], "wvs_trust_pct": [22.0, 55.0]})
    with pytest.raises(ValueError, match="holdout leakage"):
        _assert_no_wvs_columns(df)
