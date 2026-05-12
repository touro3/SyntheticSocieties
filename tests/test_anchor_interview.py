"""Tests for the /anchor and /interview endpoints.

Covers:
  - Anchor returns 404 for unknown experiment
  - Anchor answers economic majority question from events.jsonl
  - Anchor does NOT treat generic nouns ('decision') as scenario options —
    the fix for the false-tally bug where 'What was the majority decision?'
    was matched against agent text containing the word 'decision'.
  - Anchor tallies 'X or Y' scenario options from collected interview responses
  - Anchor tally count grows as more agents are interviewed
  - Interview returns 404 for unknown experiment
  - Interview (rule-based replay) answers a strategy question from events.jsonl
  - Interview appends to interview_responses.jsonl on each call
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event(agent_id: str, round_id: int, action: str, wealth: float, reasoning: str = "") -> dict:
    return {
        "agent_id": agent_id,
        "round_id": round_id,
        "action": {"action_type": action, "reasoning_summary": reasoning},
        "state_after": {"wealth": wealth, "stress": 0.1},
    }


def _write_events(exp_dir: Path, events: list[dict]) -> None:
    with (exp_dir / "events.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _write_scenario(exp_dir: Path, title: str, description: str = "") -> None:
    (exp_dir / "scenario.json").write_text(
        json.dumps({"scenario_title": title, "scenario_description": description, "population_narrative": ""})
    )


@pytest.fixture()
def app_and_root(tmp_path):
    app = create_app(experiments_root=str(tmp_path), configs_root=str(tmp_path))
    app.config["TESTING"] = True
    return app, tmp_path


@pytest.fixture()
def client(app_and_root):
    app, _ = app_and_root
    return app.test_client()


@pytest.fixture()
def root(app_and_root):
    _, root = app_and_root
    return root


# ---------------------------------------------------------------------------
# /anchor — basic routing
# ---------------------------------------------------------------------------


class TestAnchorRouting:
    def test_returns_404_for_missing_experiment(self, client):
        resp = client.post("/anchor/no_such_exp", json={"question": "What happened?"})
        assert resp.status_code == 404

    def test_returns_400_for_missing_question(self, client, root):
        exp = root / "exp_q"
        exp.mkdir()
        _write_events(exp, [_make_event("agent_0", 1, "work", 80)])
        resp = client.post("/anchor/exp_q", json={})
        assert resp.status_code == 400

    def test_returns_404_when_events_missing(self, client, root):
        exp = root / "exp_noev"
        exp.mkdir()
        resp = client.post("/anchor/exp_noev", json={"question": "What happened?"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /anchor — economic majority path
# ---------------------------------------------------------------------------


class TestAnchorEconomicMajority:
    def test_majority_question_returns_dominant_action(self, client, root):
        exp = root / "exp_maj"
        exp.mkdir()
        events = [
            _make_event("agent_0", 1, "work", 80),
            _make_event("agent_0", 2, "work", 88),
            _make_event("agent_1", 1, "cooperate", 75),
            _make_event("agent_1", 2, "work", 83),
        ]
        _write_events(exp, events)

        resp = client.post("/anchor/exp_maj", json={"question": "What was the majority action?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "work" in data["response"].lower()

    def test_majority_question_uses_economic_handler_not_scenario_tally(self, client, root):
        """'What was the majority decision?' must NOT tally 'decision' as a scenario option."""
        exp = root / "exp_decision"
        exp.mkdir()
        # Reasoning texts intentionally contain the word 'decision' to expose the old bug
        events = [
            _make_event("agent_0", 1, "work", 80, "I made a decision to work hard."),
            _make_event("agent_1", 1, "cooperate", 75, "My decision was to cooperate."),
            _make_event("agent_2", 1, "work", 82, "Hard decision but chose work."),
        ]
        _write_events(exp, events)

        resp = client.post("/anchor/exp_decision", json={"question": "What was the majority decision?"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Must NOT say "preferred 'decision'" — that's the false-tally bug
        assert "preferred 'decision'" not in data["response"]
        # Must answer with an actual economic action
        assert any(w in data["response"].lower() for w in ("work", "cooperate", "save", "steal", "action", "round"))


# ---------------------------------------------------------------------------
# /anchor — scenario opinion tally (X or Y)
# ---------------------------------------------------------------------------


class TestAnchorScenarioTally:
    def _setup_exp(self, root: Path, exp_name: str) -> Path:
        exp = root / exp_name
        exp.mkdir()
        events = [_make_event(f"agent_{i}", 1, "work", 80 + i) for i in range(5)]
        _write_events(exp, events)
        _write_scenario(exp, "Research Paper Decision Making", "Choose to write a paper or monograph.")
        return exp

    def test_or_pattern_triggers_tally(self, client, root):
        """'monograph or paper' must activate the scenario tally path."""
        exp = self._setup_exp(root, "exp_or")
        # Pre-populate interview responses so the anchor has data to tally
        responses = [
            {"agent_id": "agent_0", "question": "q", "response": "I chose to write a paper."},
            {"agent_id": "agent_1", "question": "q", "response": "I decided on a monograph."},
            {"agent_id": "agent_2", "question": "q", "response": "Writing a paper made more sense."},
        ]
        with (exp / "interview_responses.jsonl").open("w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")

        resp = client.post("/anchor/exp_or", json={"question": "monograph or paper?"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Tally should identify 'paper' as the majority (2 vs 1)
        assert "paper" in data["response"].lower()
        assert data["source"] == "anchor_interviews"

    def test_tally_count_grows_with_more_interviews(self, client, root):
        """n_interviewed in the response must reflect all collected responses."""
        exp = self._setup_exp(root, "exp_grow")

        # First batch: 3 agents interviewed
        batch1 = [
            {"agent_id": f"agent_{i}", "question": "q", "response": "I chose to write a paper."} for i in range(3)
        ]
        with (exp / "interview_responses.jsonl").open("w") as f:
            for r in batch1:
                f.write(json.dumps(r) + "\n")

        resp1 = client.post("/anchor/exp_grow", json={"question": "monograph or paper?"})
        data1 = resp1.get_json()
        assert "3 of 5" in data1["response"]

        # Second batch: 2 more agents interviewed (append)
        batch2 = [
            {"agent_id": f"agent_{i}", "question": "q", "response": "I chose to write a paper."} for i in range(3, 5)
        ]
        with (exp / "interview_responses.jsonl").open("a") as f:
            for r in batch2:
                f.write(json.dumps(r) + "\n")

        resp2 = client.post("/anchor/exp_grow", json={"question": "monograph or paper?"})
        data2 = resp2.get_json()
        assert "all 5" in data2["response"]

    def test_no_or_pattern_does_not_tally(self, client, root):
        """A plain question without 'X or Y' must not trigger the scenario tally."""
        exp = self._setup_exp(root, "exp_no_or")
        # Stored question also has no 'X or Y' pattern — no inference possible
        responses = [
            {"agent_id": "agent_0", "question": "what did you decide", "response": "I wrote a paper."},
        ]
        with (exp / "interview_responses.jsonl").open("w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")

        resp = client.post("/anchor/exp_no_or", json={"question": "What did most agents choose?"})
        assert resp.status_code == 200
        data = resp.get_json()
        # No tally source — should be regular anchor_data
        assert data["source"] != "anchor_interviews"

    def test_vague_majority_question_infers_options_from_interview_log(self, client, root):
        """'What was the majority decision?' must resolve to scenario options inferred
        from stored interview questions when agents were asked 'monograph or paper?'.

        Crucially: 'monograph' and 'paper' deliberately do NOT appear in the scenario
        title/description or in any event reasoning text — this validates that the
        inference bypasses the corpus filter and works purely from the interview question.
        """
        exp = root / "exp_infer"
        exp.mkdir()
        # Events with NO mention of monograph/paper in reasoning
        events = [_make_event(f"agent_{i}", 1, "work", 80 + i, "focused on productivity") for i in range(5)]
        _write_events(exp, events)
        # Scenario title/description also has no 'monograph' or 'paper'
        _write_scenario(exp, "Academic Career Simulation", "Researchers make career choices.")

        responses = [
            {"agent_id": "agent_0", "question": "monograph or paper?", "response": "I chose to write a paper."},
            {"agent_id": "agent_1", "question": "monograph or paper?", "response": "I decided on a monograph."},
            {"agent_id": "agent_2", "question": "monograph or paper?", "response": "Writing a paper made more sense."},
        ]
        with (exp / "interview_responses.jsonl").open("w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")

        resp = client.post("/anchor/exp_infer", json={"question": "What was the majority decision?"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Anchor must infer 'paper' as majority from interview log, not return economic actions
        assert "preferred 'decision'" not in data["response"]
        assert "paper" in data["response"].lower()
        assert data["source"] == "anchor_interviews"


# ---------------------------------------------------------------------------
# /interview — basic routing
# ---------------------------------------------------------------------------


class TestInterviewRouting:
    def test_returns_404_for_missing_experiment(self, client):
        resp = client.post("/interview/no_such_exp/agent_0", json={"question": "What happened?"})
        assert resp.status_code == 404

    def test_returns_404_when_no_events_for_agent(self, client, root):
        exp = root / "exp_int_empty"
        exp.mkdir()
        _write_events(exp, [_make_event("agent_9", 1, "work", 80)])
        # agent_0 has no events
        resp = client.post("/interview/exp_int_empty/agent_0", json={"question": "What happened?"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /interview — rule-based replay
# ---------------------------------------------------------------------------


class TestInterviewReplay:
    def _setup(self, root: Path, agent_id: str = "agent_0", n_rounds: int = 5, action: str = "work") -> Path:
        exp = root / "exp_interview"
        exp.mkdir(exist_ok=True)
        events = [_make_event(agent_id, r + 1, action, 70 + r * 3) for r in range(n_rounds)]
        _write_events(exp, events)
        return exp

    def test_strategy_question_returns_dominant_action(self, client, root):
        self._setup(root)
        resp = client.post("/interview/exp_interview/agent_0", json={"question": "What was your strategy?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "work" in data["response"].lower()
        # Source is replay_data without OPENAI_API_KEY, replay_llm when key is available
        assert data["source"] in ("replay_data", "replay_llm")

    def test_wealth_question_returns_wealth_info(self, client, root):
        self._setup(root)
        resp = client.post("/interview/exp_interview/agent_0", json={"question": "How is your wealth?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert any(w in data["response"].lower() for w in ("wealth", "rich", "comfortable", "stretched"))

    def test_interview_appends_to_response_log(self, client, root):
        exp = self._setup(root)
        log_path = exp / "interview_responses.jsonl"
        assert not log_path.exists()

        client.post("/interview/exp_interview/agent_0", json={"question": "How are you doing?"})

        assert log_path.exists()
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        assert records[0]["agent_id"] == "agent_0"

    def test_multiple_interviews_accumulate_in_log(self, client, root):
        exp = self._setup(root)
        log_path = exp / "interview_responses.jsonl"

        client.post("/interview/exp_interview/agent_0", json={"question": "First question?"})
        client.post("/interview/exp_interview/agent_0", json={"question": "Second question?"})

        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(records) == 2
