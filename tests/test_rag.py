from pathlib import Path

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
    assert "Context:" in ctx


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


def test_graph_rag_initialized_by_kernel_execution(tmp_path):
    policy = CooperativePolicy()
    agents = [
        _make_agent("agent_0", policy),
        _make_agent("agent_1", policy),
    ]

    network_manager = NetworkManager.fully_connected(
        [agent.profile.agent_id for agent in agents]
    )
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
