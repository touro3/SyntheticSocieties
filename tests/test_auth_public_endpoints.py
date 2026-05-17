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


import pytest


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/simulate-wizard", {}),
        ("/human-eval/rating", {}),
        ("/human-game/session", {}),
        ("/human-game/action", {}),
        ("/human-game/complete", {}),
    ],
)
def test_public_demo_posts_not_token_gated(monkeypatch, path, payload):
    """H1/H2: participant/demo POSTs stay reachable without a bearer token."""
    client = _client(monkeypatch)

    resp = client.post(path, json=payload)

    assert resp.status_code != 401
    body = resp.get_json(silent=True) or {}
    assert body.get("error") != "Authorization header required"


def test_human_game_session_table_is_bounded(monkeypatch):
    """H2: /human-game/session is rate-limited under a burst (DoS bound)."""
    client = _client(monkeypatch)
    statuses = {client.post("/human-game/session", json={"pre_trust": 5, "pre_risk": 5}).status_code for _ in range(50)}
    # Either created (201) or rate-limited (429) — never an auth/500 failure,
    # and the rate limiter must engage under a burst.
    assert statuses <= {201, 429}
    assert 429 in statuses


def test_simulation_config_bounds():
    """H3: oversized rounds/population_size are rejected, not silently run."""
    from pydantic import ValidationError

    from configs.schema import SimulationConfig

    with pytest.raises(ValidationError):
        SimulationConfig(rounds=10**9)
    with pytest.raises(ValidationError):
        SimulationConfig(population_size=10**9)
    # Mid-range values still validate.
    ok = SimulationConfig(rounds=100, population_size=500)
    assert ok.rounds == 100 and ok.population_size == 500


def test_redact_secrets_masks_credentials():
    """H4: redact_secrets scrubs api_key/token before snapshot persistence."""
    from utils.io import redact_secrets

    cfg = {
        "llm": {"model_id": "x", "api_key": "sk-proj-SECRET", "temperature": 0.7},
        "project": {"name": "bgf", "auth_token": "abc123"},
        "nested": [{"password": "hunter2"}],
    }
    red = redact_secrets(cfg)

    assert red["llm"]["api_key"] == "***REDACTED***"
    assert red["project"]["auth_token"] == "***REDACTED***"
    assert red["nested"][0]["password"] == "***REDACTED***"
    # Non-secret values and the original dict are untouched.
    assert red["llm"]["model_id"] == "x"
    assert cfg["llm"]["api_key"] == "sk-proj-SECRET"


def test_scrub_secrets_masks_env_keys(monkeypatch):
    """M2: subprocess log scrubber masks live key values and token shapes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-LIVEKEY12345")
    import api.app as m

    out = m._scrub_secrets("traceback: key=sk-proj-LIVEKEY12345 and gsk_abcd1234efgh")
    assert "sk-proj-LIVEKEY12345" not in out
    assert "gsk_abcd1234efgh" not in out
    assert "***REDACTED***" in out
