"""Shared test fixtures for the BGF test suite."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for all tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState

# ── Reusable factory helpers ──────────────────────────────────────────────
# how to run: pytest tests/conftest.py -v


def make_profile(**overrides) -> AgentProfile:
    """Create an AgentProfile with sensible defaults; override any field."""
    defaults = dict(
        agent_id="agent_0",
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="urban",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
        trust_people=0.5,
        competitiveness=0.5,
    )
    defaults.update(overrides)
    return AgentProfile(**defaults)


def make_state(**overrides) -> AgentState:
    """Create an AgentState with sensible defaults."""
    defaults = dict(wealth=100.0, stress=0.0, satisfaction=0.5)
    defaults.update(overrides)
    return AgentState(**defaults)


def make_agent(agent_id: str = "agent_0", wealth: float = 50.0, **profile_kw) -> Agent:
    """Create a fully wired Agent with mock policy."""
    return Agent(
        profile=make_profile(agent_id=agent_id, **profile_kw),
        state=make_state(wealth=wealth),
        memory=MemoryBuffer(max_items=10),
        policy=MockPolicy(),
    )


# ── Pytest fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_profile():
    return make_profile()


@pytest.fixture
def sample_state():
    return make_state()


@pytest.fixture
def sample_agent():
    return make_agent()


@pytest.fixture
def agent_pair():
    """Two agents suitable for interaction tests."""
    return [make_agent("agent_0"), make_agent("agent_1")]


@pytest.fixture
def world():
    return World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
    )


@pytest.fixture
def event_logger(tmp_path):
    return EventLogger(tmp_path / "events.jsonl", overwrite=True)
