"""
GraphRAG Layer using NetworkX.
Maintains a global social graph of agent interactions to provide social context to agents.
"""

import networkx as nx
import pandas as pd
from pathlib import Path
import json
from typing import Optional

class GraphRAG:

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._initialized = False

    def build_from_events(self, events_path: str | Path):
        """Construct the social graph from interaction events. Resets existing state."""
        self.graph = nx.MultiDiGraph()  # Reset to prevent duplication
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

    def add_event(self, event: dict):
        """Incrementally add a single event to the social graph."""
        if event.get("action", {}).get("action_type") == "cooperate":
            source = event["agent_id"]
            target = event["action"].get("target_agent_id")
            if source and target:
                self.graph.add_edge(
                    source, target, 
                    weight=1.0, 
                    round=event.get("round_id", 0), 
                    type="cooperate"
                )
                self._initialized = True

    def get_social_context(self, agent_id: str, k_neighbors: int = 2) -> str:
        """
        Provide a summary of the agent's position in the social network.
        """
        if not self._initialized:
            return "Social network not yet initialized."
        if agent_id not in self.graph:
            return f"Agent '{agent_id}' has no recorded interactions."

        # Outgoing/Incoming edges
        outgoing = list(self.graph.out_edges(agent_id, data=True))
        incoming = list(self.graph.in_edges(agent_id, data=True))
        
        social_summary = []
        
        if incoming:
            donors = [edge[0] for edge in incoming]
            social_summary.append(f"You have received support from: {', '.join(set(donors))}.")
        else:
            social_summary.append("You have not received any cooperation from others yet.")
            
        if outgoing:
            recipients = [edge[1] for edge in outgoing]
            social_summary.append(f"You have previously cooperated with: {', '.join(set(recipients))}.")
            
        # Social status via Centrality
        centrality = nx.degree_centrality(self.graph)
        score = centrality.get(agent_id, 0)
        
        if score > 0.5:
            social_summary.append("You are a central figure in this community.")
        elif score > 0.1:
            social_summary.append("You have established several social ties.")
        
        # k-hop context (Phase 15 addition)
        if k_neighbors > 1:
            try:
                # Find all reachable nodes within k hops
                reachable = set(nx.single_source_shortest_path_length(
                    self.graph, agent_id, cutoff=k_neighbors
                ).keys()) - {agent_id}
                if reachable:
                    social_summary.append(
                        f"You are within {k_neighbors} hops of {len(reachable)} potential collaborators."
                    )
            except nx.NetworkXError:
                pass
            
        return " ".join(social_summary)

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
            
        # Reciprocity check
        a_to_b = self.graph.number_of_edges(agent_a, agent_b)
        b_to_a = self.graph.number_of_edges(agent_b, agent_a)
        
        if a_to_b > 0 and b_to_a > 0:
            return f"Mutually beneficial relationship ({a_to_b} vs {b_to_a} interactions)."
        elif a_to_b > 0:
            return f"You have helped them {a_to_b} times, but they haven't reciprocated yet."
        elif b_to_a > 0:
            return f"They have helped you {b_to_a} times. You might owe them."
        
        return "No direct interactions recorded."

