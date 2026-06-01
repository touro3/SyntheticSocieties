from environment.network import NetworkManager


def test_fully_connected_network():
    agent_ids = ["a1", "a2", "a3"]
    network = NetworkManager.fully_connected(agent_ids)

    assert network.num_nodes() == 3
    assert network.num_edges() == 3
    assert set(network.get_neighbors("a1")) == {"a2", "a3"}


def test_random_graph_network():
    agent_ids = ["a1", "a2", "a3", "a4"]
    network = NetworkManager.random_graph(agent_ids, edge_prob=0.5, seed=42)

    assert network.num_nodes() == 4
    assert isinstance(network.get_neighbors("a1"), list)


def test_small_world_network():
    agent_ids = ["a1", "a2", "a3", "a4", "a5"]
    network = NetworkManager.small_world(agent_ids, k=2, rewiring_prob=0.3, seed=42)

    assert network.num_nodes() == 5
    assert isinstance(network.get_neighbors("a1"), list)
