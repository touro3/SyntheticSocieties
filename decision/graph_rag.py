"""GraphRAG layer using NetworkX.

Maintains an incrementally-updated directed social graph of agent cooperation
events. Provides social context (centrality, relationship history, k-hop reach)
to agents at inference time.

Performance:
  Betweenness centrality is O(VE) — expensive to recompute every round for
  every agent. This module caches the full centrality computation and only
  invalidates the cache when the graph topology changes (i.e., when a new
  cooperation edge is added). Non-structural events (work, save) do not
  invalidate the cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx


class GraphRAG:

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._initialized: bool = False

        # Centrality cache — invalidated only when graph topology changes.
        self._centrality_cache: Optional[dict] = None
        self._cache_dirty: bool = True

    # ── Graph construction ────────────────────────────────────────────────────

    def build_from_events(self, events_path: str | Path) -> None:
        """Construct the social graph from a saved events file. Resets existing state.

        Not used in the live simulation pipeline — the graph is built incrementally
        via add_event() called from RoundProcessor._update_graph_rag() after each
        round. This method is reserved for checkpoint replay: if a long run is
        interrupted, the graph can be reconstructed from the saved events.jsonl
        before resuming.
        """
        self.graph = nx.MultiDiGraph()
        self._initialized = False
        self._invalidate_cache()

        if not Path(events_path).exists():
            return

        with open(events_path, "r") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    self.add_event(event)
                except (json.JSONDecodeError, KeyError):
                    continue

        self._initialized = True

    def add_event(self, event: dict) -> None:
        """Incrementally add a single event to the social graph.

        Only cooperation events change the graph topology. Work/save events
        are ignored — this keeps the cache stable for non-social actions.
        """
        if event.get("action", {}).get("action_type") != "cooperate":
            return

        source = event.get("agent_id")
        target = event.get("action", {}).get("target_agent_id")

        if source and target:
            self.graph.add_edge(
                source, target,
                weight=1.0,
                round=event.get("round_id", 0),
                type="cooperate",
            )
            self._initialized = True
            self._invalidate_cache()

    # ── Centrality (cached) ───────────────────────────────────────────────────

    def _invalidate_cache(self) -> None:
        self._cache_dirty = True

    def _compute_centrality(self) -> dict:
        """Compute and cache degree + betweenness centrality."""
        return {
            "degree": nx.degree_centrality(self.graph),
            "betweenness": nx.betweenness_centrality(self.graph),
        }

    def _get_centrality(self) -> dict:
        """Return cached centrality, recomputing only when the graph has changed."""
        if self._cache_dirty or self._centrality_cache is None:
            self._centrality_cache = self._compute_centrality()
            self._cache_dirty = False
        return self._centrality_cache

    # ── Social context retrieval ──────────────────────────────────────────────

    def get_social_context(self, agent_id: str, k_neighbors: int = 2) -> str:
        """Return a natural-language summary of the agent's social position."""
        if not self._initialized:
            return "Social network not yet initialized."
        if agent_id not in self.graph:
            return f"Agent '{agent_id}' has no recorded interactions."

        parts: list[str] = []

        # Incoming / outgoing edge summaries
        incoming = list(self.graph.in_edges(agent_id, data=True))
        outgoing = list(self.graph.out_edges(agent_id, data=True))

        if incoming:
            donors = list({edge[0] for edge in incoming})
            parts.append(f"You have received support from: {', '.join(donors)}.")
        else:
            parts.append("You have not received any cooperation from others yet.")

        if outgoing:
            recipients = list({edge[1] for edge in outgoing})
            parts.append(f"You have previously cooperated with: {', '.join(recipients)}.")

        # Cached centrality scores
        centrality = self._get_centrality()
        degree_score = centrality["degree"].get(agent_id, 0.0)
        betweenness_score = centrality["betweenness"].get(agent_id, 0.0)

        if degree_score > 0.5:
            parts.append("You are a central figure in this community.")
        elif degree_score > 0.1:
            parts.append("You have established several social ties.")

        if betweenness_score > 0.1:
            parts.append("You act as a crucial bridge connecting different social groups.")

        # k-hop reachability
        if k_neighbors > 1:
            try:
                reachable = (
                    set(
                        nx.single_source_shortest_path_length(
                            self.graph, agent_id, cutoff=k_neighbors
                        ).keys()
                    )
                    - {agent_id}
                )
                if reachable:
                    parts.append(
                        f"You are within {k_neighbors} hops of "
                        f"{len(reachable)} potential collaborators."
                    )
            except nx.NetworkXError:
                pass

        return " ".join(parts)

    def query_relationships(self, agent_a: str, agent_b: str) -> str:
        """Query the relationship between two specific agents."""
        a_exists = agent_a in self.graph
        b_exists = agent_b in self.graph

        if not a_exists and not b_exists:
            return "Neither agent has any recorded interactions."
        if not a_exists:
            return f"Agent '{agent_a}' has no recorded interactions."
        if not b_exists:
            return f"Agent '{agent_b}' has no recorded interactions."

        a_to_b = self.graph.number_of_edges(agent_a, agent_b)
        b_to_a = self.graph.number_of_edges(agent_b, agent_a)

        if a_to_b > 0 and b_to_a > 0:
            return f"Mutually beneficial relationship ({a_to_b} vs {b_to_a} interactions)."
        elif a_to_b > 0:
            return f"You have helped them {a_to_b} times, but they haven't reciprocated yet."
        elif b_to_a > 0:
            return f"They have helped you {b_to_a} times. You might owe them."

        return "No direct interactions recorded."
