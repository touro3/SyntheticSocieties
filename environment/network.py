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

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a new edge or strengthen an existing one.

        Called by RoundProcessor when cooperation occurs, enabling
        dynamic network evolution — new cooperative relationships
        create new edges in the social graph.
        """
        if self.graph.has_edge(source, target):
            self.strengthen_edge(source, target)
        else:
            self.graph.add_edge(source, target, weight=weight)

    def strengthen_edge(self, source: str, target: str, increment: float = 0.1) -> None:
        """Increase weight of an existing edge (repeated cooperation)."""
        if self.graph.has_edge(source, target):
            current = self.graph[source][target].get("weight", 1.0)
            self.graph[source][target]["weight"] = current + increment

    def get_edge_weight(self, source: str, target: str) -> float:
        """Return the weight of an edge, or 0.0 if no edge exists."""
        if self.graph.has_edge(source, target):
            return self.graph[source][target].get("weight", 1.0)
        return 0.0

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.graph.number_of_edges()

