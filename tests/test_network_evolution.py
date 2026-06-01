"""Tests for scripts/plot_network_evolution.py — graph construction and metrics."""

import json

import networkx as nx

from scripts.plot_network_evolution import (
    build_cumulative_graphs,
    compute_metrics_over_rounds,
    load_cooperation_events,
)

# ── load_cooperation_events ──────────────────────────────────────────────────


class TestLoadCooperationEvents:
    def test_filters_cooperate_only(self, tmp_path):
        events = [
            {"round_id": 1, "agent_id": "a0", "action": {"action_type": "work", "target_agent_id": None}},
            {"round_id": 1, "agent_id": "a1", "action": {"action_type": "cooperate", "target_agent_id": "a0"}},
            {"round_id": 2, "agent_id": "a0", "action": {"action_type": "save", "target_agent_id": None}},
            {"round_id": 2, "agent_id": "a2", "action": {"action_type": "cooperate", "target_agent_id": "a1"}},
        ]
        path = tmp_path / "events.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in events))
        result = load_cooperation_events(path)
        assert len(result) == 2
        assert result[0]["agent_id"] == "a1"
        assert result[1]["target_id"] == "a1"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text("")
        result = load_cooperation_events(path)
        assert result == []

    def test_cooperate_without_target_ignored(self, tmp_path):
        events = [
            {"round_id": 1, "agent_id": "a0", "action": {"action_type": "cooperate", "target_agent_id": None}},
        ]
        path = tmp_path / "events.jsonl"
        path.write_text(json.dumps(events[0]))
        result = load_cooperation_events(path)
        assert result == []


# ── build_cumulative_graphs ──────────────────────────────────────────────────


class TestBuildCumulativeGraphs:
    def test_cumulative_edges(self):
        events = [
            {"round_id": 1, "agent_id": "a0", "target_id": "a1"},
            {"round_id": 2, "agent_id": "a1", "target_id": "a2"},
        ]
        graphs = build_cumulative_graphs(events, {"a0", "a1", "a2"}, max_round=2)
        assert graphs[1].number_of_edges() == 1
        assert graphs[2].number_of_edges() == 2

    def test_repeated_edge_increases_weight(self):
        events = [
            {"round_id": 1, "agent_id": "a0", "target_id": "a1"},
            {"round_id": 2, "agent_id": "a0", "target_id": "a1"},
        ]
        graphs = build_cumulative_graphs(events, {"a0", "a1"}, max_round=2)
        assert graphs[2].number_of_edges() == 1
        assert graphs[2]["a0"]["a1"]["weight"] == 2

    def test_all_agents_present_as_nodes(self):
        events = [{"round_id": 1, "agent_id": "a0", "target_id": "a1"}]
        graphs = build_cumulative_graphs(events, {"a0", "a1", "a2", "a3"}, max_round=1)
        assert graphs[1].number_of_nodes() == 4

    def test_empty_events_produces_edgeless_graphs(self):
        graphs = build_cumulative_graphs([], {"a0", "a1"}, max_round=3)
        for r, G in graphs.items():
            assert G.number_of_edges() == 0
            assert G.number_of_nodes() == 2


# ── compute_metrics_over_rounds ──────────────────────────────────────────────


class TestComputeMetricsOverRounds:
    def test_basic_metrics(self):
        events = [
            {"round_id": 1, "agent_id": "a0", "target_id": "a1"},
            {"round_id": 2, "agent_id": "a1", "target_id": "a2"},
            {"round_id": 3, "agent_id": "a0", "target_id": "a2"},
        ]
        graphs = build_cumulative_graphs(events, {"a0", "a1", "a2"}, max_round=3)
        metrics = compute_metrics_over_rounds(graphs, [1, 2, 3])
        assert metrics["round"] == [1, 2, 3]
        assert metrics["n_edges"] == [1, 2, 3]
        # Density should increase
        assert metrics["density"][2] > metrics["density"][0]

    def test_components_decrease_with_edges(self):
        events = [
            {"round_id": 1, "agent_id": "a0", "target_id": "a1"},
            {"round_id": 2, "agent_id": "a2", "target_id": "a3"},
            {"round_id": 3, "agent_id": "a1", "target_id": "a2"},
        ]
        graphs = build_cumulative_graphs(events, {"a0", "a1", "a2", "a3"}, max_round=3)
        metrics = compute_metrics_over_rounds(graphs, [1, 2, 3])
        # Round 1: {a0,a1} + isolated a2,a3 → 3 components
        # Round 2: {a0,a1} + {a2,a3} → 2 components
        # Round 3: all connected → 1 component
        assert metrics["n_components"][0] == 3
        assert metrics["n_components"][1] == 2
        assert metrics["n_components"][2] == 1

    def test_missing_round_skipped(self):
        graphs = {1: nx.Graph(), 3: nx.Graph()}
        graphs[1].add_nodes_from(["a0", "a1"])
        graphs[3].add_nodes_from(["a0", "a1"])
        metrics = compute_metrics_over_rounds(graphs, [1, 2, 3])
        # Round 2 missing from graphs → skipped
        assert metrics["round"] == [1, 3]
