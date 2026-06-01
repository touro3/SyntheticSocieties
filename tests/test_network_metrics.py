"""Tests for network structure metrics."""

import networkx as nx

from metrics.network_metrics import assortativity, diffusion_speed, modularity, network_summary


def test_assortativity_complete_graph():
    G = nx.complete_graph(10)
    # Complete graph has undefined/zero assortativity
    a = assortativity(G)
    assert -1 <= a <= 1


def test_assortativity_empty_graph():
    G = nx.Graph()
    assert assortativity(G) == 0.0


def test_modularity_complete_graph():
    G = nx.complete_graph(10)
    m = modularity(G)
    assert -0.5 <= m <= 1.0


def test_modularity_two_cliques():
    """Two disconnected cliques should have high modularity."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])  # clique 1
    G.add_edges_from([(3, 4), (3, 5), (4, 5)])  # clique 2
    m = modularity(G)
    assert m > 0.3, f"Two cliques should have high modularity, got {m}"


def test_modularity_empty():
    G = nx.Graph()
    assert modularity(G) == 0.0


def test_diffusion_speed_complete():
    G = nx.complete_graph(10)
    d = diffusion_speed(G)
    assert d == 1.0, f"Complete graph has avg path 1, speed = 1.0, got {d}"


def test_diffusion_speed_path():
    G = nx.path_graph(10)
    d = diffusion_speed(G)
    assert 0 < d < 1.0, f"Path graph should have speed < 1, got {d}"


def test_diffusion_speed_disconnected():
    """Should use largest connected component."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])
    G.add_node(3)  # isolated
    d = diffusion_speed(G)
    assert d > 0


def test_diffusion_speed_trivial():
    G = nx.Graph()
    G.add_node(0)
    assert diffusion_speed(G) == 0.0


def test_network_summary_structure():
    G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
    s = network_summary(G)
    assert "n_nodes" in s
    assert "n_edges" in s
    assert "density" in s
    assert "assortativity" in s
    assert "modularity" in s
    assert "diffusion_speed" in s
    assert "avg_degree" in s
    assert "clustering_coefficient" in s
    assert s["n_nodes"] == 20
