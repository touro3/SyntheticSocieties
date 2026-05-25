"""Extract H4 cooperation-graph Newman modularity from the §8.1 cells.

Reads `events.jsonl` for every `experiments/mx_{A,B}_s*/` cell that matches
the given suffix, builds the cumulative cooperation graph via the existing
`scripts.plot_network_evolution.build_cumulative_graphs` helper, computes
Newman modularity Q via `metrics.network_metrics.modularity`, and writes
one CSV row per cell. Re-uses existing utilities — no duplicated logic.

Usage:
    python scripts/extract_modularity_from_cells.py            # default: mx_{A,B}_s{1..10}
    python scripts/extract_modularity_from_cells.py --suffix n500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metrics.network_metrics import modularity  # noqa: E402


def _build_cooperation_graph(events_path: Path) -> tuple[nx.Graph, int]:
    """Cumulative cooperation graph weighted by edge frequency.

    Returns (G, max_round). Reads only `cooperate` actions with a non-null
    target_agent_id — same predicate as
    scripts/plot_network_evolution.load_cooperation_events."""
    edge_w: dict[tuple[str, str], int] = defaultdict(int)
    all_agents: set[str] = set()
    max_round = 0
    with events_path.open() as f:
        for line in f:
            e = json.loads(line)
            aid = e["agent_id"]
            all_agents.add(aid)
            max_round = max(max_round, int(e.get("round_id", 0)))
            act = e.get("action") or {}
            if act.get("action_type") == "cooperate":
                tgt = act.get("target_agent_id")
                if tgt:
                    all_agents.add(tgt)
                    edge_w[tuple(sorted((aid, tgt)))] += 1
    G = nx.Graph()
    G.add_nodes_from(all_agents)
    for (a, b), w in edge_w.items():
        G.add_edge(a, b, weight=w)
    return G, max_round


def extract(suffix: str | None, seeds: range, conditions: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for cond in conditions:
        for seed in seeds:
            cell_id = f"mx_{cond}_{suffix}_s{seed}" if suffix else f"mx_{cond}_s{seed}"
            ev_path = REPO_ROOT / "experiments" / cell_id / "events.jsonl"
            if not ev_path.exists():
                print(f"  skip {cell_id} (no events.jsonl)")
                continue
            G, max_round = _build_cooperation_graph(ev_path)
            Q = modularity(G) if G.number_of_edges() > 0 else 0.0
            rows.append(
                {
                    "cell": cell_id,
                    "condition": cond,
                    "seed": seed,
                    "n_nodes": G.number_of_nodes(),
                    "n_edges": G.number_of_edges(),
                    "density": round(nx.density(G), 6),
                    "n_components": nx.number_connected_components(G),
                    "modularity_Q": round(Q, 6),
                    "max_round": max_round,
                }
            )
            print(f"  {cell_id}: edges={G.number_of_edges():5d} Q={Q:.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suffix", default="", help="Cell-id suffix (e.g. 'n500'); default: §8.1 N=100 cells")
    ap.add_argument("--conditions", default="A,B", help="Comma-separated condition list (default: A,B)")
    ap.add_argument("--seeds", default="1-10", help="Seed range, e.g. '1-10' or '1,2,3'")
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "analysis" / "tables" / "network_modularity_8_1.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        seeds = range(lo, hi + 1)
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    suffix = args.suffix or None

    rows = extract(suffix=suffix, seeds=seeds, conditions=conditions)
    if not rows:
        print("ERROR: no cells found — verify the suffix / seeds / conditions match disk layout.")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
