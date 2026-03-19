import pytest
from pathlib import Path
import json
import networkx as nx
from decision.graph_rag import GraphRAG
from decision.sql_rag import SQLRAG

def test_graph_rag_reset():
    rag = GraphRAG()
    # Mock event
    event = {
        "agent_id": "a1",
        "action": {"action_type": "cooperate", "target_agent_id": "a2"},
        "round_id": 1
    }
    rag.add_event(event)
    assert rag.graph.number_of_edges() == 1
    
    # Test reset on build_from_events (even if file is empty)
    p = Path("/tmp/empty_events.jsonl")
    p.write_text("")
    rag.build_from_events(p)
    assert rag.graph.number_of_edges() == 0

def test_graph_rag_centrality():
    rag = GraphRAG()
    # Star topology: a1 helps everyone
    for i in range(10):
        rag.add_event({
            "agent_id": "a1",
            "action": {"action_type": "cooperate", "target_agent_id": f"a{i}"}
        })
    
    ctx = rag.get_social_context("a1")
    assert "central figure" in ctx

def test_sql_rag_security():
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    
    # Injection attempt in query_population_trends
    res = rag.query_population_trends("DROP TABLE population")
    assert "Only SELECT queries are permitted" in res

def test_sql_rag_parameterization():
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    # This should work regardless of input types
    ctx = rag.get_peer_group_context(age=30, gender="female")
    assert "Context:" in ctx

def test_sql_rag_missing_data():
    rag = SQLRAG(data_path="data/ess_clean.parquet")
    # Impossible age
    ctx = rag.get_peer_group_context(age=500, gender=1)
    assert "No peer group data found" in ctx
