from __future__ import annotations

import networkx as nx


class NetworkManager:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    @classmethod
    def fully_connected(cls, agent_ids: list[str]) -> NetworkManager:
        graph = nx.Graph()
        graph.add_nodes_from(agent_ids)

        for i, source in enumerate(agent_ids):
            for target in agent_ids[i + 1:]:
                graph.add_edge(source, target)

        return cls(graph)

    @classmethod
    def random_graph(cls, agent_ids: list[str], edge_prob: float, seed: int | None = None) -> NetworkManager:
        graph = nx.erdos_renyi_graph(n=len(agent_ids), p=edge_prob, seed=seed)
        relabeled = cls._relabel_graph(graph, agent_ids)
        return cls(relabeled)

    @classmethod
    def small_world(
        cls,
        agent_ids: list[str],
        k: int,
        rewiring_prob: float,
        seed: int | None = None,
    ) -> NetworkManager:
        graph = nx.watts_strogatz_graph(
            n=len(agent_ids),
            k=k,
            p=rewiring_prob,
            seed=seed,
        )
        relabeled = cls._relabel_graph(graph, agent_ids)
        return cls(relabeled)

    @staticmethod
    def _relabel_graph(graph: nx.Graph, agent_ids: list[str]) -> nx.Graph:
        """Relabel integer-indexed NetworkX graph to use agent_id strings."""
        mapping = {i: agent_ids[i] for i in range(len(agent_ids))}
        return nx.relabel_nodes(graph, mapping)

    def get_neighbors(self, agent_id: str) -> list[str]:
        if agent_id not in self.graph:
            return []
        return list(self.graph.neighbors(agent_id))

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.graph.number_of_edges()
