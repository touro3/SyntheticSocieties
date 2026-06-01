"""BGF_DATA_ROOT reroutes user-write paths under a single env-configurable root.

Local dev (env unset) must keep the historical layout; HF Space deploys with
BGF_DATA_ROOT=/data must reroute experiments, uploads, the tracker index,
and human-eval outputs onto persistent storage.
"""

import importlib

import pytest


def _reload_app_module():
    import api.app as app_mod

    return importlib.reload(app_mod)


def test_default_paths_unchanged_when_env_unset(monkeypatch):
    monkeypatch.delenv("BGF_DATA_ROOT", raising=False)
    app_mod = _reload_app_module()
    assert app_mod._EXPERIMENTS_ROOT.as_posix() == "experiments"
    assert app_mod._UPLOADS_DIR.as_posix() == "uploads/ess_data"
    assert app_mod._TRACKER_INDEX.as_posix() == "tracker/experiment_index.parquet"
    assert app_mod._HUMAN_OUTPUTS_DIR.as_posix() == "data/human"


def test_paths_reroute_under_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    app_mod = _reload_app_module()
    assert app_mod._EXPERIMENTS_ROOT == tmp_path / "experiments"
    assert app_mod._UPLOADS_DIR == tmp_path / "uploads" / "ess_data"
    assert app_mod._TRACKER_INDEX == tmp_path / "tracker" / "experiment_index.parquet"
    assert app_mod._HUMAN_OUTPUTS_DIR == tmp_path / "human_outputs"


def test_create_app_mkdirs_under_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BGF_API_TOKEN", "")
    app_mod = _reload_app_module()
    app_mod.create_app()
    assert (tmp_path / "experiments").is_dir()
    assert (tmp_path / "uploads" / "ess_data").is_dir()
    assert (tmp_path / "tracker").is_dir()
    assert (tmp_path / "human_outputs").is_dir()


def test_health_does_not_leak_key_presence(tmp_path, monkeypatch):
    """Public /health must not advertise which provider keys are configured."""
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BGF_API_TOKEN", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fakefake")
    app_mod = _reload_app_module()
    client = app_mod.create_app().test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "checks" in body
    checks = body["checks"]
    assert "openai_key" not in checks
    assert "groq_key" not in checks


def test_admin_health_requires_token(tmp_path, monkeypatch):
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BGF_API_TOKEN", "super-secret-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    app_mod = _reload_app_module()
    client = app_mod.create_app().test_client()

    # No bearer → 401.
    assert client.get("/admin/health").status_code == 401
    # Wrong bearer → 401.
    assert client.get("/admin/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    # Correct bearer → 200 + key flags.
    resp = client.get("/admin/health", headers={"Authorization": "Bearer super-secret-token"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["checks"]["openai_key"] is True


def test_admin_health_disabled_in_open_mode(tmp_path, monkeypatch):
    """Without BGF_API_TOKEN, /admin/health must 404 — never serve key presence."""
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BGF_API_TOKEN", "")
    app_mod = _reload_app_module()
    client = app_mod.create_app().test_client()
    resp = client.get("/admin/health")
    assert resp.status_code == 404
