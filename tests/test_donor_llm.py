"""Phase 2 — LLM donor-decision bridge + API backend message conversion (no API)."""

from __future__ import annotations

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.anthropic_backend import _split_system
from decision.donor_llm import build_donor_prompt, make_llm_donor_decider, parse_donor_response
from decision.gemini_backend import _to_gemini
from decision.mock_policy import MockPolicy
from environment.donor_game import DONATE, KEEP, ReputationLedger, ReputationView


def _agent(aid="a0", trust=0.7, country="AT"):
    return Agent(
        profile=AgentProfile(
            agent_id=aid,
            age=35,
            income=1000.0,
            education="secondary",
            occupation="worker",
            location="urban",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
            trust_people=trust,
            country=country,
        ),
        state=AgentState(wealth=0.0),
        memory=HierarchicalMemory(max_recent=10),
        policy=MockPolicy(),
    )


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_parse_keywords():
    assert parse_donor_response("DONATE, they earned it") == DONATE
    assert parse_donor_response("keep — bad reputation") == KEEP
    assert parse_donor_response("Donate") == DONATE


def test_parse_defaults_to_keep():
    assert parse_donor_response("") == KEEP
    assert parse_donor_response("undecided") == KEEP  # Nash fallback


# ── Prompt construction ───────────────────────────────────────────────────────


def test_prompt_includes_reputation_and_strategy():
    donor = _agent()
    donor.inherited_strategy = "Reward good reputations."
    rep = ReputationView("r1", [DONATE, DONATE, KEEP], 2 / 3, 3)
    msgs = build_donor_prompt(donor, _agent("r1"), rep, grounded=False)
    user = msgs[-1]["content"]
    assert "67%" in user or "66%" in user
    assert "Reward good reputations." in user
    # Ungrounded: no persona leak.
    assert "trust-in-others" not in user


def test_prompt_first_encounter_phrasing():
    rep = ReputationLedger().view("new")
    msgs = build_donor_prompt(_agent(), _agent("new"), rep, grounded=False)
    assert "first encounter" in msgs[-1]["content"].lower()


def test_grounded_prompt_includes_persona():
    rep = ReputationLedger().view("new")
    msgs = build_donor_prompt(_agent(trust=0.81), _agent("new"), rep, grounded=True)
    assert "trust-in-others 0.81" in msgs[-1]["content"]


# ── Decider with a fake backend ───────────────────────────────────────────────


class _FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, messages, temperature=None):
        self.calls += 1
        return self.reply, 0.0


def test_make_llm_donor_decider_parses_backend_reply():
    decider = make_llm_donor_decider(_FakeBackend("KEEP\nthey have a poor image"))
    rep = ReputationView("r1", [KEEP], 0.0, 1)
    decision = decider(_agent(), _agent("r1"), rep, 0)
    assert decision.action == KEEP
    assert "poor image" in decision.reasoning


def test_decider_donate_path():
    fake = _FakeBackend("DONATE")
    decider = make_llm_donor_decider(fake)
    decision = decider(_agent(), _agent("r1"), ReputationLedger().view("r1"), 0)
    assert decision.action == DONATE
    assert fake.calls == 1


# ── Backend message conversion ────────────────────────────────────────────────


def test_anthropic_split_system():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    system, convo = _split_system(msgs)
    assert system == "sys"
    assert convo == [{"role": "user", "content": "hi"}]


def test_anthropic_split_requires_user_turn():
    system, convo = _split_system([{"role": "system", "content": "only system"}])
    assert convo and convo[0]["role"] == "user"


def test_gemini_role_mapping():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ho"},
    ]
    system, contents = _to_gemini(msgs)
    assert system == "sys"
    assert [c["role"] for c in contents] == ["user", "model"]
    assert contents[0]["parts"][0]["text"] == "hi"
