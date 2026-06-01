"""Interview persona + memory + scenario context, and anchor stance extraction."""

import importlib
import json


def _reload_app(monkeypatch, tmp_path):
    monkeypatch.setenv("BGF_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("BGF_API_TOKEN", "")
    import api.app as app_mod

    return importlib.reload(app_mod)


def _write_exp(tmp_path, exp_id, *, snapshot=None, config_yaml=None):
    exp_dir = tmp_path / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        with (exp_dir / "population_snapshot.jsonl").open("w") as f:
            for rec in snapshot:
                f.write(json.dumps(rec) + "\n")
    if config_yaml is not None:
        (exp_dir / "config.yaml").write_text(config_yaml)
    return exp_dir


def _mk_events(n_rounds, action_pattern, *, neighbors=("agent_5",), target="agent_5"):
    events = []
    for r in range(1, n_rounds + 1):
        events.append(
            {
                "round_id": r,
                "agent_id": "agent_0",
                "action": {
                    "action_type": action_pattern(r),
                    "target_agent_id": target if action_pattern(r) in {"cooperate", "steal"} else None,
                    "reasoning_summary": f"r{r} reasoning",
                },
                "state_after": {"wealth": 50 + r, "stress": min(1.0, 0.1 + r * 0.01)},
                "perception": {"network": {"neighbors": list(neighbors)}},
            }
        )
    return events


def test_interview_helper_renders_persona_block(monkeypatch, tmp_path):
    app_mod = _reload_app(monkeypatch, tmp_path)
    exp_dir = _write_exp(
        tmp_path,
        "exp1",
        snapshot=[
            {
                "agent_id": "agent_0",
                "age": 42,
                "gender": 1,
                "country": "AT",
                "trust_people": 0.7,
                "trust_institutions": 0.3,
                "risk_tolerance": 0.45,
                "competitiveness": 0.65,
                "left_right": 0.55,
            }
        ],
        config_yaml=(
            "simulation:\n  rounds: 10\n  population_size: 5\n"
            "policy:\n  type: rule_based\n"
            "network:\n  type: random\n"
            "population:\n  source: empirical\n"
        ),
    )
    events = _mk_events(10, lambda r: "cooperate" if r % 2 == 0 else "save")
    ctx = app_mod._interview_prompt_context(exp_dir, "agent_0", events)
    assert "42-year-old" in ctx["persona_block"]
    assert "from AT" in ctx["persona_block"]
    # High interpersonal trust + low institutional trust must appear distinctly.
    assert "high interpersonal trust" in ctx["persona_block"]
    assert "low institutional trust" in ctx["persona_block"]
    # Scenario block always renders, even without scenario.json.
    assert "10-round" in ctx["scenario_block"]
    assert ctx["history_window"] >= 10
    assert "Repeated cooperation partners" in ctx["social_block"]


def test_interview_helper_degrades_without_snapshot(monkeypatch, tmp_path):
    app_mod = _reload_app(monkeypatch, tmp_path)
    exp_dir = _write_exp(tmp_path, "exp_no_snap")
    events = _mk_events(5, lambda r: "save")
    ctx = app_mod._interview_prompt_context(exp_dir, "agent_0", events)
    assert ctx["persona_block"] == ""
    # Scenario block still rendered (from defaults), not empty.
    assert ctx["scenario_block"]


def test_interview_history_window_adapts_to_run_length(monkeypatch, tmp_path):
    app_mod = _reload_app(monkeypatch, tmp_path)
    exp_dir = _write_exp(tmp_path, "exp_long")
    # 90 rounds → window = min(30, max(10, 90//3)) = 30
    events = _mk_events(90, lambda r: "work")
    ctx = app_mod._interview_prompt_context(exp_dir, "agent_0", events)
    assert ctx["history_window"] == 30
    # 12 rounds → window = max(10, 12//3) = 10
    events = _mk_events(12, lambda r: "work")
    ctx = app_mod._interview_prompt_context(exp_dir, "agent_0", events)
    assert ctx["history_window"] == 10


def test_interview_includes_provenance_in_response(monkeypatch, tmp_path):
    """The /interview JSON response includes persona_used/history_window/source.

    No OpenAI key is configured here, so the response comes from the
    rule-based fallback — but it must still carry the new provenance fields.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app_mod = _reload_app(monkeypatch, tmp_path)
    exp_id = "wizard_1"
    exp_dir = _write_exp(
        tmp_path,
        exp_id,
        snapshot=[{"agent_id": "agent_0", "age": 30, "country": "DE", "trust_people": 0.5}],
        config_yaml="simulation:\n  rounds: 5\n  population_size: 5\n",
    )
    events = _mk_events(5, lambda r: "save")
    with (exp_dir / "events.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    client = app_mod.create_app().test_client()
    resp = client.post(f"/interview/{exp_id}/agent_0", json={"question": "What did you do?"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "persona_used" in body
    assert body["persona_used"] is True
    assert body["history_window"] == 10
    assert body["source"] == "replay_data"
    assert body["fallback_reason"] == "no_openai_key"


def test_stance_extractor_cache_dedup(monkeypatch, tmp_path):
    """The stance cache must return the same result for identical inputs."""
    app_mod = _reload_app(monkeypatch, tmp_path)

    # Inject a fake client to count calls.
    calls = []

    class _FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})

    class _FakeResp:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _FakeResp(json.dumps({"stance": "paper", "rationale": "preferred", "topic_tags": ["preference"]}))

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeOAI:
        def __init__(self, **kw):
            self.chat = _FakeChat()

    # Replace the OAI factory cache so _extract_stance_via_llm sees our fake.
    monkeypatch.setitem(app_mod._OAI_CLIENTS, "testkey0", _FakeOAI())

    # Reset stance cache so this test does not see leftovers.
    app_mod._STANCE_CACHE.clear()

    result1 = app_mod._extract_stance_via_llm(
        "I really prefer paper", "paper or monograph?", ["paper", "monograph"], api_key="testkey0_value"
    )
    result2 = app_mod._extract_stance_via_llm(
        "I really prefer paper", "paper or monograph?", ["paper", "monograph"], api_key="testkey0_value"
    )
    assert result1 is not None and result1["stance"] == "paper"
    assert result2 is not None and result2["stance"] == "paper"
    # Second call should be served from cache, not invoke OpenAI again.
    assert len(calls) == 1


def test_semantic_stance_tally_falls_back_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app_mod = _reload_app(monkeypatch, tmp_path)
    out = app_mod._semantic_stance_tally({"agent_0": "I prefer paper"}, "paper or monograph?", ["paper", "monograph"])
    assert out is None  # Caller falls back to substring tally
