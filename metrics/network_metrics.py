"""
Network structure metrics for simulation analysis.

Implements assortativity, modularity, and diffusion speed
as specified in the BGF Phase 8 evaluation metrics.
"""

from __future__ import annotations

import warnings
import math
import networkx as nx
import numpy as np


def assortativity(G: nx.Graph) -> float:
    """
    Compute degree assortativity coefficient.

    Returns
    -------
    float
        Degree assortativity in [-1, 1] when defined.
        Returns 0.0 for degenerate graphs where assortativity is undefined
        (e.g., empty graphs, complete graphs, or zero-variance degree sequences).

    Notes
    -----
    NetworkX may emit a RuntimeWarning for graphs where the assortativity
    denominator is zero. We treat those cases explicitly as 0.0 so the
    metric pipeline remains numerically stable and reproducible.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in scalar divide",
                category=RuntimeWarning,
            )
            value = nx.degree_assortativity_coefficient(G)

        if value is None or not np.isfinite(value):
            return 0.0

        # defensive clipping against tiny floating-point spillover
        return float(np.clip(value, -1.0, 1.0))

    except ZeroDivisionError:
        return 0.0


def modularity(G: nx.Graph) -> float:
    """
    Compute modularity score using greedy community detection.

    Higher values indicate stronger community structure.

    Returns:
        Modularity in [-0.5, 1]. Returns 0.0 for empty graphs.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0

    communities = nx.community.greedy_modularity_communities(G)
    return float(nx.community.modularity(G, communities))


def diffusion_speed(G: nx.Graph) -> float:
    """
    Compute diffusion speed as inverse of average shortest path length.

    Higher values = faster information diffusion.
    For disconnected graphs, uses the largest connected component.

    Returns:
        Diffusion speed (positive float). Returns 0.0 for trivial graphs.
    """
    if G.number_of_nodes() < 2:
        return 0.0

    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) < 2:
            return 0.0
        subgraph = G.subgraph(largest_cc)
        avg_path = nx.average_shortest_path_length(subgraph)

    return float(1.0 / avg_path) if avg_path > 0 else 0.0


def network_summary(G: nx.Graph) -> dict:
    """
    Compute a comprehensive network structure summary.

    Returns:
        Dict with n_nodes, n_edges, density, assortativity, modularity, diffusion_speed.
    """
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": float(nx.density(G)),
        "assortativity": assortativity(G),
        "modularity": modularity(G),
        "diffusion_speed": diffusion_speed(G),
        "avg_degree": float(2 * G.number_of_edges() / G.number_of_nodes()) if G.number_of_nodes() > 0 else 0.0,
        "clustering_coefficient": float(nx.average_clustering(G)) if G.number_of_nodes() > 0 else 0.0,
    }
