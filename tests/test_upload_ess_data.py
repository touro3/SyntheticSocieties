"""Upload-endpoint hardening: format support + schema validation."""

import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from api.app import _load_uploaded_dataframe, create_app
from population.column_aliases import COLUMN_ALIASES, normalize_columns


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Sandbox writes so test runs cannot pollute the repo's experiments/.
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    # Explicit empty value blocks dotenv (override=False) from re-loading a
    # token from the repo's .env, keeping the test in open mode.
    monkeypatch.setenv("BGF_API_TOKEN", "")
    # Force re-import so module-level data-root constants pick up the new env.
    import importlib

    import api.app as app_mod

    importlib.reload(app_mod)
    return app_mod.create_app().test_client()


def test_csv_comma_delimited(client):
    csv = b"age,gender,country,trust_people\n42,1,AT,0.7\n50,2,DE,0.4\n"
    resp = client.post(
        "/upload-ess-data",
        data={"file": (io.BytesIO(csv), "ess.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["rows"] == 2


def test_csv_semicolon_delimited(client):
    csv = b"age;gender;country;income_decile\n42;1;AT;5\n50;2;DE;-99\n"
    resp = client.post(
        "/upload-ess-data",
        data={"file": (io.BytesIO(csv), "ess.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    # The -99 sentinel should be coerced to NaN, so completeness < 100%.
    assert body["analysis"]["quality"]["completeness_pct"] < 100.0


def test_ess_short_codes_aliased(client):
    csv = (
        "agea,gndr,cntry,trstprl,ppltrst,stflife,lrscale\n"
        "42,1,AT,5,6,7,3\n"
        "50,2,DE,6,5,8,5\n"
    )
    resp = client.post(
        "/upload-ess-data",
        data={"file": (io.BytesIO(csv.encode()), "ess_round11.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    alias_hits = body["analysis"]["normalization"]["alias_hits"]
    assert alias_hits["agea"] == "age"
    assert alias_hits["cntry"] == "country"
    assert alias_hits["ppltrst"] == "trust_people"
    # Renamed columns must include canonical names.
    assert "age" in body["columns"]
    assert "country" in body["columns"]


def test_missing_age_returns_structured_422(client):
    df = pd.DataFrame({"gender": [1, 2], "country": ["AT", "DE"]})
    buf = io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)
    resp = client.post(
        "/upload-ess-data",
        data={"file": (buf, "no_age.parquet")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 422
    body = resp.get_json()
    assert "missing_required" in body and "age" in body["missing_required"]
    assert "agea" in body["aliases_tried"]["age"]


def test_unsupported_extension_rejected(client):
    resp = client.post(
        "/upload-ess-data",
        data={"file": (io.BytesIO(b"x"), "foo.xlsx")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_sav_graceful_when_pyreadstat_missing():
    # Direct call to the loader so we test the fallback message even if
    # pyreadstat is installed in CI — we monkey-patch the import to fail.
    import sys

    backup = sys.modules.pop("pyreadstat", None)
    sys.modules["pyreadstat"] = None  # makes `import pyreadstat` raise TypeError
    try:
        df, err = _load_uploaded_dataframe("test.sav", b"fake")
        assert df is None
        assert err is not None and "pyreadstat" in err.lower()
    finally:
        if backup is not None:
            sys.modules["pyreadstat"] = backup
        else:
            sys.modules.pop("pyreadstat", None)


def test_dta_round_trip():
    df_in = pd.DataFrame({"age": [40, 50], "gender": [1, 2]})
    buf = io.BytesIO()
    df_in.to_stata(buf, write_index=False)
    df, err = _load_uploaded_dataframe("test.dta", buf.getvalue())
    assert err is None
    assert df.shape == (2, 2)


def test_normalize_columns_handles_collisions():
    # edulvla and eisced both map to education_level; only the first wins.
    rename, _ = normalize_columns(["edulvla", "eisced"])
    assert rename == {"edulvla": "education_level"}


def test_normalize_columns_case_insensitive_no_alias_hit():
    # Pure case normalization is not surfaced as an alias hit — it's noise.
    _rename, hits = normalize_columns(["AGE", "Country"])
    assert hits == {}
