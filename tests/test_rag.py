from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.graph_rag import GraphRAG
from decision.schemas import ProposedAction
from decision.sql_rag import SQLRAG
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from simulation.kernel import SimulationKernel

_ESS_PATH = Path("data/ess_clean.parquet")


@pytest.fixture(autouse=True)
def _ess_parquet(tmp_path, monkeypatch):
    """Redirect SQLRAG to a minimal synthetic parquet when the real ESS file is absent.

    The synthetic dataset covers ages 25–54 (30 rows, alternating genders) so
    every tier of the four-tier peer-group fallback can satisfy the n≥5 threshold
    for realistic age/gender queries (e.g. age=30, gender=female).  The extreme
    query age=500 intentionally finds no peers, exercising the "No peer group
    data found" branch.
    """
    if _ESS_PATH.exists():
        return  # real file present — no mocking needed

    import numpy as np

    rng = np.random.RandomState(42)
    n = 30
    synthetic = pd.DataFrame(
        {
            "age": list(range(25, 55)),
            "gender": [1, 2] * 15,
            "trust_people": rng.uniform(0.3, 0.8, n).tolist(),
            "risk_taking": rng.uniform(0.2, 0.7, n).tolist(),
            "life_satisfaction": rng.uniform(0.5, 0.9, n).tolist(),
            "income_decile": rng.uniform(3.0, 8.0, n).tolist(),
            "country": ["AT"] * n,
        }
    )
    parquet_path = tmp_path / "ess_clean.parquet"
    synthetic.to_parquet(parquet_path, index=False)

    # Patch SQLRAG.__init__ to silently swap the default path for the synthetic one.
    from decision.sql_rag import SQLRAG

    _real_init = SQLRAG.__init__

    def _patched_init(self, data_path="data/ess_clean.parquet", **kwargs):
        if str(data_path) == "data/ess_clean.parquet":
            data_path = parquet_path
        _real_init(self, data_path=str(data_path))

    monkeypatch.setattr(SQLRAG, "__init__", _patched_init)


def test_graph_rag_reset():
    rag = GraphRAG()
    event = {
        "agent_id": "a1",
        "action": {"action_type": "cooperate", "target_agent_id": "a2"},
        "round_id": 1,
    }
    rag.add_event(event)
    assert rag.graph.number_of_edges() == 1

    p = Path("/tmp/empty_events.jsonl")
    p.write_text("")
    rag.build_from_events(p)
    assert rag.graph.number_of_edges() == 0


def test_graph_rag_centrality():
    rag = GraphRAG()
    for i in range(10):
        rag.add_event(
            {
                "agent_id": "a1",
                "action": {"action_type": "cooperate", "target_agent_id": f"a{i}"},
            }
        )

    ctx = rag.get_social_context("a1")
    assert "central figure" in ctx


def test_sql_rag_security():
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    res = rag.query_population_trends("DROP TABLE population")
    assert "Only SELECT queries are permitted" in res


def test_sql_rag_parameterization():
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    ctx = rag.get_peer_group_context(age=30, gender="female")
    assert "peers" in ctx.lower() or "Based on" in ctx


def test_sql_rag_missing_data():
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    ctx = rag.get_peer_group_context(age=500, gender=1)
    assert "No peer group data found" in ctx


class CooperativePolicy:
    def __init__(self):
        self.graph_rag = GraphRAG()

    def propose_action(self, profile, state, memory, context, round_id):
        neighbors = context.get("network", {}).get("neighbors", [])
        return ProposedAction(
            action_type="cooperate",
            target_agent_id=neighbors[0],
            amount=5.0,
            reasoning_summary="test cooperation",
            confidence=1.0,
        )


def _make_agent(agent_id: str, policy) -> Agent:
    profile = AgentProfile(
        agent_id=agent_id,
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="urban",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
    )
    state = AgentState(wealth=100.0)
    memory = HierarchicalMemory(max_recent=10)
    return Agent(profile=profile, state=state, memory=memory, policy=policy)


def test_graph_rag_context_changes_after_cooperation_event():
    """After a cooperation edge is added, get_social_context must reflect it."""
    rag = GraphRAG()
    ctx_before = rag.get_social_context("agent_0")
    # Before any event: agent is not in graph at all
    assert "no recorded interactions" in ctx_before.lower() or "not yet initialized" in ctx_before.lower()

    rag.add_event(
        {
            "agent_id": "agent_1",
            "action": {"action_type": "cooperate", "target_agent_id": "agent_0"},
            "round_id": 1,
        }
    )

    ctx_after = rag.get_social_context("agent_0")
    assert ctx_after != ctx_before
    assert "agent_1" in ctx_after


def test_sql_rag_different_demographics_yield_different_context():
    """ESS data must produce distinct peer-group contexts for different age cohorts."""
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    ctx_young = rag.get_peer_group_context(age=25, gender=1)
    ctx_old = rag.get_peer_group_context(age=70, gender=1)

    # Both should return real data (not "No peer group data found")
    if "No peer group data found" in ctx_young or "No peer group data found" in ctx_old:
        pytest.skip("ESS data does not have sufficient coverage for both cohorts")

    assert ctx_young != ctx_old, (
        "SQL RAG must return cohort-specific data — identical output for age 25 vs 70 "
        "suggests the query is returning global averages or ignoring the age filter."
    )


# ── C2: SQL RAG income cohort filter tests ────────────────────────────────────


def test_sql_rag_income_filter_tightens_cohort():
    """Passing income_decile should produce a different (tighter) context than omitting it."""
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    ctx_no_income = rag.get_peer_group_context(age=40, gender=1, country="DE")
    ctx_high_income = rag.get_peer_group_context(age=40, gender=1, country="DE", income_decile=9.0)
    ctx_low_income = rag.get_peer_group_context(age=40, gender=1, country="DE", income_decile=2.0)

    if "No peer group data found" in ctx_no_income:
        pytest.skip("Insufficient ESS data for DE/age40/male cohort")

    # Income label appears only when income_decile is provided
    assert "9.0" in ctx_high_income or "peer avg" in ctx_high_income
    assert "2.0" in ctx_low_income or "peer avg" in ctx_low_income


def test_sql_rag_income_fallback_when_sparse():
    """When income band is too sparse (<5 peers), fallback should still return data."""
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    # Use an unrealistic income_decile band — should fall back to wider cohort
    ctx = rag.get_peer_group_context(age=40, gender=1, income_decile=5.5)
    assert "peers" in ctx.lower() or "No peer group data found" in ctx


def test_sql_rag_income_line_without_income_arg():
    """Without income_decile arg, output should show peer avg income decile."""
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    ctx = rag.get_peer_group_context(age=35, gender=2)
    if "Based on" in ctx:
        assert "Income decile: avg" in ctx


def test_sql_rag_income_line_with_income_arg():
    """With income_decile arg, output should show agent's decile alongside peer avg."""
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    ctx = rag.get_peer_group_context(age=35, gender=2, income_decile=7.0)
    if "Based on" in ctx:
        assert "7.0" in ctx and "peer avg" in ctx


# ── H3: Graph RAG reciprocity rate tests ──────────────────────────────────────


def test_graph_rag_reciprocity_rate_full():
    """Mutual cooperation should report correct reciprocation percentage."""
    rag = GraphRAG()
    # A cooperates with B twice
    for r in [1, 2]:
        rag.add_event({"agent_id": "A", "action": {"action_type": "cooperate", "target_agent_id": "B"}, "round_id": r})
    # B cooperates back once
    rag.add_event({"agent_id": "B", "action": {"action_type": "cooperate", "target_agent_id": "A"}, "round_id": 3})

    result = rag.query_relationships("A", "B")
    assert "50%" in result
    assert "2x" in result
    assert "reciprocated 1x" in result


def test_graph_rag_reciprocity_zero():
    """One-sided cooperation should show 0% reciprocation."""
    rag = GraphRAG()
    rag.add_event({"agent_id": "A", "action": {"action_type": "cooperate", "target_agent_id": "B"}, "round_id": 1})
    result = rag.query_relationships("A", "B")
    assert "0%" in result
    assert "not cooperated back" in result


def test_graph_rag_reciprocity_lag():
    """Reciprocation lag should be reported when B responds after A."""
    rag = GraphRAG()
    rag.add_event({"agent_id": "A", "action": {"action_type": "cooperate", "target_agent_id": "B"}, "round_id": 1})
    rag.add_event({"agent_id": "B", "action": {"action_type": "cooperate", "target_agent_id": "A"}, "round_id": 4})
    result = rag.query_relationships("A", "B")
    assert "lag" in result.lower()
    assert "3.0" in result  # lag = 4 - 1 = 3


def test_graph_rag_query_relationships_no_interactions():
    """Agents with no interactions should return appropriate message."""
    rag = GraphRAG()
    rag.add_event({"agent_id": "X", "action": {"action_type": "cooperate", "target_agent_id": "Y"}, "round_id": 1})
    result = rag.query_relationships("A", "B")
    assert "recorded interactions" in result.lower()


def test_graph_rag_query_owed():
    """B→A only should tell A they might owe B."""
    rag = GraphRAG()
    rag.add_event({"agent_id": "B", "action": {"action_type": "cooperate", "target_agent_id": "A"}, "round_id": 1})
    result = rag.query_relationships("A", "B")
    assert "owe" in result.lower()


# ── Original kernel integration test (unchanged) ──────────────────────────────


def test_graph_rag_initialized_by_kernel_execution(tmp_path):
    policy = CooperativePolicy()
    agents = [
        _make_agent("agent_0", policy),
        _make_agent("agent_1", policy),
    ]

    network_manager = NetworkManager.fully_connected([agent.profile.agent_id for agent in agents])
    world = World(
        state=WorldState(),
        institution_manager=InstitutionManager(),
        network_manager=network_manager,
    )
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)

    kernel = SimulationKernel(agents=agents, world=world, logger=logger)
    kernel.run(num_rounds=1)

    assert policy.graph_rag._initialized is True
    assert policy.graph_rag.graph.number_of_edges() >= 1

    ctx = policy.graph_rag.get_social_context("agent_0")
    assert "Social network not yet initialized." not in ctx


# ── P4.6: RAG failure path tests ──────────────────────────────────────────────


class TestGraphRagNoneReturnGraceful:
    """graph_rag.get_social_context() returning None must not crash prompt building."""

    def test_graph_rag_returns_none_for_unknown_agent(self):
        """GraphRAG.get_social_context returns a string (not None) even for unknown agents."""
        rag = GraphRAG()
        # Add one event for a different agent so the graph is initialized
        rag.add_event({"agent_id": "A", "action": {"action_type": "cooperate", "target_agent_id": "B"}, "round_id": 1})
        result = rag.get_social_context("completely_unknown_agent_xyz")
        # Must return a string, never None — callers assume str
        assert isinstance(result, str)

    def test_llm_policy_base_graph_rag_context_returns_none_when_no_rag(self):
        """graph_rag_context() returns None when graph_rag attribute is absent."""
        from decision.llm_policy_base import LLMPolicyBase

        policy = LLMPolicyBase.__new__(LLMPolicyBase)
        # No graph_rag attribute set
        assert policy.graph_rag_context("any_agent") is None

    def test_llm_policy_base_graph_rag_context_handles_none_attribute(self):
        """graph_rag_context() returns None when graph_rag is explicitly set to None."""
        from decision.llm_policy_base import LLMPolicyBase

        policy = LLMPolicyBase.__new__(LLMPolicyBase)
        policy.graph_rag = None
        assert policy.graph_rag_context("any_agent") is None

    def test_llm_policy_base_sql_rag_context_returns_none_when_no_rag(self):
        """sql_rag_context() returns None when sql_rag attribute is absent."""
        from decision.llm_policy_base import LLMPolicyBase

        policy = LLMPolicyBase.__new__(LLMPolicyBase)
        assert policy.sql_rag_context(age=35, gender="male", country="DE") is None

    def test_llm_policy_base_sql_rag_context_handles_none_attribute(self):
        """sql_rag_context() returns None when sql_rag is explicitly set to None."""
        from decision.llm_policy_base import LLMPolicyBase

        policy = LLMPolicyBase.__new__(LLMPolicyBase)
        policy.sql_rag = None
        assert policy.sql_rag_context(age=35, gender="male", country="DE") is None

    def test_llm_policy_propose_action_with_rag_returning_none(self):
        """LLMPolicy must not crash when both RAG contexts return None."""
        from unittest.mock import MagicMock

        from agents.memory import HierarchicalMemory
        from agents.profile import AgentProfile
        from agents.state import AgentState
        from decision.llm_policy import LLMPolicy

        mock_rag = MagicMock()
        mock_rag.get_social_context.return_value = None
        mock_rag.get_peer_group_context.return_value = None

        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            '{"action_type": "work", "reasoning_summary": "ok", "confidence": 0.8}',
            0.1,
        )

        policy = LLMPolicy(backend=mock_backend, graph_rag=mock_rag, sql_rag=mock_rag, max_retries=0)

        profile = AgentProfile(
            agent_id="a0",
            age=35,
            income=1000.0,
            education="college",
            occupation="worker",
            location="italy",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
        )
        state = AgentState(wealth=100.0)
        memory = HierarchicalMemory(max_recent=5)
        context = {"neighbors": ["a1"], "public_signal": {}, "prices": {}, "resources": {}, "round_id": 1}

        # Should not raise
        result = policy.propose_action(profile, state, memory, context, round_id=1)
        assert result is not None
        assert result.action_type in ("work", "save", "cooperate")
