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

    # Temporal decay: edges older than this many rounds lose salience.
    _EDGE_HALF_LIFE = 10

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._initialized: bool = False
        self._current_round: int = 0

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
        round_id = event.get("round_id", 0)

        if source and target:
            self.graph.add_edge(
                source, target,
                weight=1.0,
                round=round_id,
                type="cooperate",
            )
            self._current_round = max(self._current_round, round_id)
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

    def _edge_recency_weight(self, edge_round: int) -> float:
        """Compute recency weight for an edge based on its round.

        Returns a value in (0, 1] where 1.0 = current round, 0.5 = half-life rounds ago.
        """
        import math
        age = max(0, self._current_round - edge_round)
        decay = math.log(2) / max(self._EDGE_HALF_LIFE, 1)
        return math.exp(-decay * age)

    def get_social_context(self, agent_id: str, k_neighbors: int = 2) -> str:
        """Return a natural-language summary of the agent's social position.

        Uses recency-weighted edge analysis so that stale relationships
        from early rounds don't dominate the social context.
        """
        if not self._initialized:
            return "Social network not yet initialized."
        if agent_id not in self.graph:
            return f"Agent '{agent_id}' has no recorded interactions."

        parts: list[str] = []

        # Incoming / outgoing edge summaries with recency filtering.
        # Only report partners with meaningful recent activity.
        incoming = list(self.graph.in_edges(agent_id, data=True))
        outgoing = list(self.graph.out_edges(agent_id, data=True))

        if incoming:
            # Weight each donor by recency and report only recent ones
            donor_weights: dict[str, float] = {}
            for src, _, data in incoming:
                w = self._edge_recency_weight(data.get("round", 0))
                donor_weights[src] = donor_weights.get(src, 0.0) + w
            # Filter to donors with meaningful recent activity (weight > 0.3)
            recent_donors = [d for d, w in sorted(donor_weights.items(), key=lambda x: -x[1]) if w > 0.3]
            if recent_donors:
                parts.append(f"You have recently received support from: {', '.join(recent_donors[:5])}.")
            else:
                all_donors = list(donor_weights.keys())[:3]
                parts.append(f"You received support in earlier rounds from: {', '.join(all_donors)} (not recent).")
        else:
            parts.append("You have not received any cooperation from others yet.")

        if outgoing:
            recipient_weights: dict[str, float] = {}
            for _, tgt, data in outgoing:
                w = self._edge_recency_weight(data.get("round", 0))
                recipient_weights[tgt] = recipient_weights.get(tgt, 0.0) + w
            recent_recipients = [r for r, w in sorted(recipient_weights.items(), key=lambda x: -x[1]) if w > 0.3]
            if recent_recipients:
                parts.append(f"You have recently cooperated with: {', '.join(recent_recipients[:5])}.")
            elif recipient_weights:
                old_recipients = list(recipient_weights.keys())[:3]
                parts.append(f"You cooperated with {', '.join(old_recipients)} in earlier rounds (not recent).")

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
        """Query the relationship between two specific agents.

        Returns interaction counts, reciprocation rate, and average lag (in rounds)
        between agent_a's cooperation and agent_b's response — a direct signal for
        the trust gradient hypothesis (high trust → fast, high reciprocation).
        """
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

        if a_to_b == 0 and b_to_a == 0:
            return "No direct interactions recorded."

        # Reciprocation rate: fraction of A→B cooperations that B later returned
        reciprocation_rate = b_to_a / a_to_b if a_to_b > 0 else 0.0

        # Average rounds to reciprocate: for each A→B at round r, find nearest
        # B→A edge at round r' >= r and compute mean(r' - r).
        avg_lag: Optional[float] = None
        if a_to_b > 0 and b_to_a > 0:
            a_rounds = sorted(
                d.get("round", 0)
                for src, tgt, d in self.graph.edges(agent_a, data=True)
                if tgt == agent_b
            )
            b_rounds = sorted(
                d.get("round", 0)
                for src, tgt, d in self.graph.edges(agent_b, data=True)
                if tgt == agent_a
            )
            lags = []
            for r_a in a_rounds:
                future = [r_b for r_b in b_rounds if r_b >= r_a]
                if future:
                    lags.append(min(future) - r_a)
            if lags:
                avg_lag = sum(lags) / len(lags)

        parts: list[str] = []
        if a_to_b > 0 and b_to_a > 0:
            pct = int(reciprocation_rate * 100)
            parts.append(
                f"Mutually beneficial relationship: you cooperated {a_to_b}x, "
                f"they reciprocated {b_to_a}x ({pct}% reciprocation rate)."
            )
            if avg_lag is not None:
                parts.append(f"Average lag to reciprocation: {avg_lag:.1f} rounds.")
        elif a_to_b > 0:
            parts.append(
                f"You have helped them {a_to_b} times with 0% reciprocation — "
                "they have not cooperated back yet."
            )
        else:
            parts.append(f"They have helped you {b_to_a} times. You might owe them.")

        return " ".join(parts)
