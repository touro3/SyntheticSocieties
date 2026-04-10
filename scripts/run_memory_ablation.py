"""
Memory Ablation Experiment Runner (Phase 29.4).

Runs 2×4 factorial design:
  - 4 memory levels: M0, M1, M2, M3
  - 2 grounding conditions: grounded (ESS/RAG) vs ungrounded (no RAG)
  - 3 seeds: 42, 123, 7
  → 24 total runs

Usage:
    # Dry-run (5 agents, 3 rounds, no GPU):
    python scripts/run_memory_ablation.py --dry-run --agents 5 --rounds 3

    # Full run (GPU required):
    python scripts/run_memory_ablation.py --seeds 42,123,7

    # Single level test:
    python scripts/run_memory_ablation.py --levels 0,1 --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MEMORY_LEVEL_NAMES = {0: "M0_no_memory", 1: "M1_window", 2: "M2_archive", 3: "M3_full"}

# Base config for ablation runs (overrides applied per condition)
BASE_OVERRIDES = {
    "policy.type": "llm",
    "population.source": "empirical",
}

# Grounding conditions
CONDITIONS = {
    "grounded":   {"policy.type": "llm"},        # RAG on (default LLMPolicy with RAG)
    "ungrounded": {"policy.type": "ablated_llm", "ablation.mode": "no_rag"},
}


def build_run_matrix(
    levels: list[int],
    seeds: list[int],
    conditions: list[str],
    agents: int,
    rounds: int,
) -> list[dict]:
    """Return list of run specification dicts."""
    runs = []
    for level in levels:
        for cond_name in conditions:
            for seed in seeds:
                level_name = MEMORY_LEVEL_NAMES[level]
                exp_id = f"ablation_{level_name}_{cond_name}_s{seed}"
                runs.append({
                    "experiment_id": exp_id,
                    "memory_level": level,
                    "condition": cond_name,
                    "seed": seed,
                    "agents": agents,
                    "rounds": rounds,
                })
    return runs


def build_overrides(run: dict, dry_run: bool) -> list[str]:
    """Build CLI override list for run_config_simulation.py."""
    overrides = [
        f"project.experiment_id={run['experiment_id']}",
        f"project.seed={run['seed']}",
        f"memory.level={run['memory_level']}",
        f"simulation.population_size={run['agents']}",
        f"simulation.rounds={run['rounds']}",
    ]
    cond_overrides = CONDITIONS[run["condition"]]
    for k, v in cond_overrides.items():
        overrides.append(f"{k}={v}")
    if dry_run:
        overrides.append("policy.type=mock")
    return overrides


def run_single(run: dict, dry_run: bool, verbose: bool) -> bool:
    """Execute a single ablation run. Returns True on success."""
    config_path = REPO_ROOT / "configs" / "memory_ablation" / f"m{run['memory_level']}_{'no_memory' if run['memory_level']==0 else 'window_only' if run['memory_level']==1 else 'window_archive' if run['memory_level']==2 else 'full'}.yaml"

    # Fall back to base config if level-specific one missing
    if not config_path.exists():
        config_path = REPO_ROOT / "configs" / "base_config.yaml"

    overrides = build_overrides(run, dry_run)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_config_simulation.py"),
           "--config", str(config_path)] + overrides

    if verbose:
        print(f"\n→ {run['experiment_id']}")
        print("  " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  FAILED: {run['experiment_id']}")
            if not verbose and result.stderr:
                print(result.stderr[-500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {run['experiment_id']}")
        return False
    except Exception as exc:
        print(f"  ERROR: {run['experiment_id']}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory ablation experiment runner.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock policy — fast, no GPU needed.")
    parser.add_argument("--agents", type=int, default=100,
                        help="Population size (default 100; use 5 for dry-run).")
    parser.add_argument("--rounds", type=int, default=30,
                        help="Simulation rounds (default 30; use 3 for dry-run).")
    parser.add_argument("--seeds", type=str, default="42,123,7",
                        help="Comma-separated seeds (default 42,123,7).")
    parser.add_argument("--levels", type=str, default="0,1,2,3",
                        help="Comma-separated memory levels to run (default all).")
    parser.add_argument("--conditions", type=str, default="grounded,ungrounded",
                        help="Comma-separated conditions (default both).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    levels = [int(l) for l in args.levels.split(",")]
    conds = [c.strip() for c in args.conditions.split(",")]

    runs = build_run_matrix(levels, seeds, conds, args.agents, args.rounds)

    print(f"Memory ablation: {len(runs)} runs planned")
    print(f"  Levels:     {levels}")
    print(f"  Conditions: {conds}")
    print(f"  Seeds:      {seeds}")
    print(f"  Dry-run:    {args.dry_run}")
    print()

    results = {"passed": [], "failed": []}
    for i, run in enumerate(runs, 1):
        label = f"[{i:2d}/{len(runs)}] {run['experiment_id']}"
        print(label, end=" ", flush=True)
        ok = run_single(run, dry_run=args.dry_run, verbose=args.verbose)
        status = "OK" if ok else "FAIL"
        print(status)
        (results["passed"] if ok else results["failed"]).append(run["experiment_id"])

    print(f"\nDone: {len(results['passed'])} passed, {len(results['failed'])} failed")
    if results["failed"]:
        print("Failed runs:")
        for r in results["failed"]:
            print(f"  {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
