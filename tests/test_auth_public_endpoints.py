"""Regression: /design-simulation is a public, rate-limited demo endpoint.

When BGF_API_TOKEN is configured, ordinary POST routes require the bearer
token, but /design-simulation must stay reachable without one (the token
would otherwise have to be shipped in client-side JS). Cost/DoS on this
endpoint is bounded by the per-IP rate limiter, not by the token.
"""

import api.app as app_module
from api.app import create_app


def _client(monkeypatch):
    monkeypatch.setattr(app_module, "_AUTH_TOKEN", "secret-token")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_design_simulation_does_not_require_token(monkeypatch):
    client = _client(monkeypatch)

    resp = client.post("/design-simulation", json={})

    # Must NOT be blocked by the auth gate. It may 400 (missing prompt) or
    # 5xx (no LLM key in test env) — anything but the auth rejection.
    assert resp.status_code != 401
    body = resp.get_json(silent=True) or {}
    assert body.get("error") != "Authorization header required"


def test_other_post_still_requires_token(monkeypatch):
    client = _client(monkeypatch)

    resp = client.post("/simulate", json={})

    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Authorization header required"
