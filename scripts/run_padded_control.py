"""Padded Prompt Control pipeline — Condition P.

Phase 28 / TOP_TIER_RESEARCH Section 1.

Runs Condition P (padded ablation) across multiple seeds. Each run uses
``PaddedAblationPolicy``, which matches Condition B token counts but
replaces ESS content with semantically neutral filler.

Usage:
    # Small test run (CPU-only, mock policy)
    python scripts/run_padded_control.py --seeds 42 --agents 10 --rounds 5

    # Full paper-quality run (GPU required)
    python scripts/run_padded_control.py --seeds 42,123,7 --agents 500 --rounds 30

    # With fixed token budget instead of per-agent dynamic measurement
    python scripts/run_padded_control.py --seeds 42,123,7 --target-tokens 450

Output:
    experiments/padded_control_s{seed}/ per seed, containing:
        events.jsonl, summary.json, metadata.json, config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config
from utils.io import ensure_dir, save_json, save_yaml, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run Condition P (padded prompt control).")
    parser.add_argument(
        "--seeds", type=str, default="42,123,7",
        help="Comma-separated random seeds (default: 42,123,7).",
    )
    parser.add_argument(
        "--agents", type=int, default=500,
        help="Population size per run (default: 500).",
    )
    parser.add_argument(
        "--rounds", type=int, default=30,
        help="Simulation rounds per run (default: 30).",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=None,
        help="Fixed target token count per padded prompt. "
             "If omitted, measured dynamically per agent to match Condition B.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/condition_p.yaml",
        help="Base config path (default: configs/condition_p.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments",
        help="Root directory for experiment output.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without running the simulation.",
    )
    return parser.parse_args()


def build_run_config(base_config: dict, seed: int, args) -> dict:
    """Merge base config with seed-specific overrides."""
    import copy
    cfg = copy.deepcopy(base_config)

    cfg["project"]["seed"] = seed
    cfg["project"]["experiment_id"] = f"padded_control_s{seed}"
    cfg["simulation"]["population_size"] = args.agents
    cfg["simulation"]["rounds"] = args.rounds

    if args.target_tokens is not None:
        cfg.setdefault("padded", {})["target_token_count"] = args.target_tokens
    else:
        cfg.setdefault("padded", {})["target_token_count"] = None

    return cfg


def run_single_seed(cfg: dict, output_dir: Path) -> dict:
    """Run a single Condition P experiment and return summary metadata."""
    from agents.agent import Agent
    from bgf_logging.event_logger import EventLogger
    from metrics.event_metrics import behavior_summary_from_events, load_events
    from metrics.summary import merge_behavior_summary, summarize_agents
    from population.generator import generate_empirical_population, generate_population
    from simulation.kernel import SimulationKernel
    from environment.institutions import InstitutionManager
    from environment.network import NetworkManager
    from environment.world import World
    from environment.world_state import WorldState
    from scripts.run_config_simulation import build_policy, build_network, build_world

    experiment_id = cfg["project"]["experiment_id"]
    seed = cfg["project"]["seed"]
    set_global_seed(seed)

    run_dir = ensure_dir(output_dir / experiment_id)
    save_yaml(cfg, run_dir / "config.yaml")

    metadata = {
        "project_name": cfg["project"]["name"],
        "experiment_id": experiment_id,
        "condition": "P",
        "seed": seed,
        "policy_type": cfg["policy"]["type"],
        "population_size": cfg["simulation"]["population_size"],
        "rounds": cfg["simulation"]["rounds"],
        "target_token_count": cfg.get("padded", {}).get("target_token_count"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(metadata, run_dir / "metadata.json")

    policy = build_policy(cfg)

    pop_source = cfg.get("population", {}).get("source", "synthetic")
    if pop_source == "empirical":
        agents = generate_empirical_population(cfg, policy)
    else:
        agents = generate_population(cfg, policy)

    print(f"  [{experiment_id}] {len(agents)} agents, {cfg['simulation']['rounds']} rounds")

    network_manager = build_network(cfg, agents)
    world = build_world(cfg, network_manager)
    logger = EventLogger(run_dir / "events.jsonl", overwrite=True)

    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=logger,
        heartbeat_path=run_dir / "heartbeat.json",
    )

    t0 = time.time()
    kernel.run(num_rounds=cfg["simulation"]["rounds"])
    elapsed = time.time() - t0

    summary = summarize_agents(agents)
    events = load_events(run_dir / "events.jsonl")
    event_behavior = behavior_summary_from_events(events)
    summary = merge_behavior_summary(summary, event_behavior)
    save_json(summary, run_dir / "summary.json")

    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = round(elapsed, 1)
    save_json(metadata, run_dir / "metadata.json")

    print(f"  [{experiment_id}] Done in {elapsed:.1f}s — saved to {run_dir}")
    return metadata


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    base_config = load_config(args.config)
    output_dir = Path(args.output_dir)

    print(f"Condition P — Padded Prompt Control")
    print(f"  Seeds:   {seeds}")
    print(f"  Agents:  {args.agents}")
    print(f"  Rounds:  {args.rounds}")
    print(f"  Tokens:  {'dynamic (per-agent)' if args.target_tokens is None else args.target_tokens}")
    print(f"  Config:  {args.config}")
    print()

    if args.dry_run:
        sample_cfg = build_run_config(base_config, seeds[0], args)
        print("Dry run — would execute:")
        for seed in seeds:
            print(f"  experiments/padded_control_s{seed}/")
        print("\nSample config (seed 0):")
        print(json.dumps(sample_cfg, indent=2, default=str))
        return

    results = []
    for seed in seeds:
        cfg = build_run_config(base_config, seed, args)
        print(f"Running seed {seed}...")
        meta = run_single_seed(cfg, output_dir)
        results.append(meta)

    print(f"\nAll {len(seeds)} Condition P runs complete.")
    print("Next step: run analyze_padded_vs_grounded.py to compare against A and B.")


if __name__ == "__main__":
    main()
