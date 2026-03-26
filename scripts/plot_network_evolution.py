"""Network evolution visualization — cooperation graph structure over rounds.

Reads events.jsonl from an experiment directory and produces:
  1. A multi-panel figure showing the cooperation graph at selected rounds.
  2. A time-series plot of network metrics (edges, density, clustering) over rounds.

Usage:
    python scripts/plot_network_evolution.py --experiment experiments/cmp_llm_s42
    python scripts/plot_network_evolution.py --experiment experiments/cmp_llm_s42 --rounds 1,5,10,20,30
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def load_cooperation_events(events_path: Path) -> list[dict]:
    """Load all events and filter to cooperation actions."""
    events = []
    with events_path.open() as f:
        for line in f:
            evt = json.loads(line)
            action = evt.get("action", {})
            if action.get("action_type") == "cooperate" and action.get("target_agent_id"):
                events.append({
                    "round_id": evt["round_id"],
                    "agent_id": evt["agent_id"],
                    "target_id": action["target_agent_id"],
                })
    return events


def build_cumulative_graphs(
    events: list[dict], all_agents: set[str], max_round: int
) -> dict[int, nx.Graph]:
    """Build a cumulative cooperation graph at each round.

    Returns {round_id: Graph} where each graph includes all cooperation
    edges up to and including that round, weighted by frequency.
    """
    edge_counts: dict[int, dict[tuple[str, str], int]] = {}
    current_counts: dict[tuple[str, str], int] = defaultdict(int)

    events_by_round: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        events_by_round[e["round_id"]].append(e)

    for r in range(1, max_round + 1):
        for e in events_by_round.get(r, []):
            edge = tuple(sorted([e["agent_id"], e["target_id"]]))
            current_counts[edge] += 1
        edge_counts[r] = dict(current_counts)

    graphs = {}
    for r, counts in edge_counts.items():
        G = nx.Graph()
        G.add_nodes_from(all_agents)
        for (a, b), w in counts.items():
            G.add_edge(a, b, weight=w)
        graphs[r] = G

    return graphs


def compute_metrics_over_rounds(
    graphs: dict[int, nx.Graph], rounds: list[int]
) -> dict[str, list[float]]:
    """Compute network metrics at each round."""
    metrics = {
        "round": [],
        "n_edges": [],
        "density": [],
        "avg_clustering": [],
        "n_components": [],
    }
    for r in sorted(rounds):
        G = graphs.get(r)
        if G is None:
            continue
        metrics["round"].append(r)
        metrics["n_edges"].append(G.number_of_edges())
        metrics["density"].append(nx.density(G))
        metrics["avg_clustering"].append(
            nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
        )
        metrics["n_components"].append(nx.number_connected_components(G))
    return metrics


def plot_snapshots(
    graphs: dict[int, nx.Graph],
    snapshot_rounds: list[int],
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """Plot cooperation graph snapshots at selected rounds."""
    n = len(snapshot_rounds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), facecolor="white")
    if n == 1:
        axes = [axes]

    # Compute a shared layout from the final graph for visual consistency
    all_rounds = sorted(graphs.keys())
    final_graph = graphs[all_rounds[-1]] if all_rounds else nx.Graph()
    pos = nx.spring_layout(final_graph, k=0.3, iterations=500, seed=42)

    for ax, r in zip(axes, snapshot_rounds):
        G = graphs.get(r)
        if G is None:
            ax.set_title(f"Round {r}\n(no data)")
            ax.axis("off")
            continue

        weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.5 + 3.0 * (w / max_w) for w in weights]

        degrees = dict(G.degree())
        node_sizes = [80 + 20 * degrees.get(n, 0) for n in G.nodes()]

        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_widths, alpha=0.4, edge_color="steelblue"
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes, node_color="coral",
            linewidths=0.5, edgecolors="gray"
        )
        ax.set_title(f"Round {r}\n({G.number_of_edges()} edges)", fontsize=13)
        ax.axis("off")

    fig.suptitle(f"{title_prefix}Cooperation Network Evolution", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_timeseries(
    metrics: dict[str, list[float]], output_path: Path, title_prefix: str = ""
) -> None:
    """Plot network metrics as a time series."""
    rounds = metrics["round"]
    if not rounds:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")

    axes[0, 0].plot(rounds, metrics["n_edges"], "o-", color="steelblue", markersize=3)
    axes[0, 0].set_ylabel("Cooperation Edges")
    axes[0, 0].set_title("Cumulative Cooperation Edges")

    axes[0, 1].plot(rounds, metrics["density"], "o-", color="coral", markersize=3)
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Graph Density")

    axes[1, 0].plot(rounds, metrics["avg_clustering"], "o-", color="seagreen", markersize=3)
    axes[1, 0].set_ylabel("Avg Clustering")
    axes[1, 0].set_title("Clustering Coefficient")

    axes[1, 1].plot(rounds, metrics["n_components"], "o-", color="mediumpurple", markersize=3)
    axes[1, 1].set_ylabel("Components")
    axes[1, 1].set_title("Connected Components")

    for ax in axes.flat:
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix}Network Metrics Over Time", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Network evolution visualization")
    parser.add_argument("--experiment", required=True, help="Path to experiment directory")
    parser.add_argument(
        "--rounds", default=None,
        help="Comma-separated round IDs for snapshot panels (default: auto-select 5)"
    )
    parser.add_argument("--output-dir", default="analysis/figures", help="Output directory")
    args = parser.parse_args()

    exp_dir = Path(args.experiment)
    events_path = exp_dir / "events.jsonl"
    if not events_path.exists():
        print(f"No events.jsonl found in {exp_dir}")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_name = exp_dir.name

    print(f"Loading events from {events_path}...")
    coop_events = load_cooperation_events(events_path)
    print(f"  Found {len(coop_events)} cooperation events")

    if not coop_events:
        print("  No cooperation events — nothing to plot.")
        return

    # Discover all agents and max round from the full events file
    all_agents = set()
    max_round = 0
    with events_path.open() as f:
        for line in f:
            evt = json.loads(line)
            all_agents.add(evt["agent_id"])
            max_round = max(max_round, evt["round_id"])

    print(f"  {len(all_agents)} agents, {max_round} rounds")

    graphs = build_cumulative_graphs(coop_events, all_agents, max_round)
    all_rounds = sorted(graphs.keys())

    if args.rounds:
        snapshot_rounds = [int(r) for r in args.rounds.split(",")]
    else:
        # Auto-select ~5 evenly spaced rounds
        indices = np.linspace(0, len(all_rounds) - 1, min(5, len(all_rounds)), dtype=int)
        snapshot_rounds = [all_rounds[i] for i in indices]

    prefix = f"{exp_name}: "
    plot_snapshots(graphs, snapshot_rounds, out_dir / f"network_evolution_{exp_name}.png", prefix)
    print(f"  Saved snapshots: {out_dir / f'network_evolution_{exp_name}.png'}")

    metrics = compute_metrics_over_rounds(graphs, all_rounds)
    plot_metric_timeseries(metrics, out_dir / f"network_metrics_{exp_name}.png", prefix)
    print(f"  Saved metrics: {out_dir / f'network_metrics_{exp_name}.png'}")


if __name__ == "__main__":
    main()
