"""Placebo / semantic-isolation ablation runner (audit response).

Runs a 3-arm contrast to isolate whether behavioral realism comes from the
*sociological coherence* of ESS grounding or merely from prompt heterogeneity:

    grounded      → population.source=empirical   (ESS-coherent personas)
    placebo       → population.source=placebo     (scrambled-but-valid control)
    unconditioned → population.source=synthetic   (config-default personas)

Design: 3 arms × N seeds. Every other factor (policy, network, rounds,
population size) is held identical, so the only varying factor is the
grounding signal. Each run produces the standard experiment artifacts
(``events.jsonl`` round-by-round trajectories, ``summary.json`` endpoint
metrics) under ``experiments/<exp_id>/`` — trajectories are logged, not just
endpoint distributions, so the thesis methodology can audit decision paths.

Usage
-----
    # Dry-run (mock policy, fast, no GPU):
    python scripts/run_placebo_ablation.py --dry-run --agents 6 --rounds 4

    # Full run (GPU required):
    python scripts/run_placebo_ablation.py --seeds 42,123,7

    # Single arm:
    python scripts/run_placebo_ablation.py --arms placebo --dry-run

Downstream analysis: ``analysis/placebo_variance.py`` reads the resulting
summaries and runs the existing one-way variance decomposition over the 3
conditions, separating the *semantic* component (grounded vs placebo) from
the *any-conditioning* component (placebo vs unconditioned).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# arm name → population.source value
ARMS = {
    "grounded": "empirical",
    "placebo": "placebo",
    "unconditioned": "synthetic",
}


def build_run_matrix(arms: list[str], seeds: list[int], agents: int, rounds: int) -> list[dict]:
    """Return list of run specification dicts (one per arm × seed)."""
    runs = []
    for arm in arms:
        for seed in seeds:
            runs.append(
                {
                    "experiment_id": f"placebo_abl_{arm}_s{seed}",
                    "arm": arm,
                    "source": ARMS[arm],
                    "seed": seed,
                    "agents": agents,
                    "rounds": rounds,
                }
            )
    return runs


def build_overrides(run: dict, dry_run: bool) -> list[str]:
    """Build CLI override list for run_config_simulation.py."""
    overrides = [
        f"project.experiment_id={run['experiment_id']}",
        f"project.seed={run['seed']}",
        f"population.source={run['source']}",
        f"simulation.population_size={run['agents']}",
        f"simulation.rounds={run['rounds']}",
    ]
    if dry_run:
        overrides.append("policy.type=mock")
    return overrides


def run_single(run: dict, dry_run: bool, verbose: bool) -> bool:
    """Execute a single ablation run. Returns True on success."""
    config_path = REPO_ROOT / "configs" / "ablation_placebo.yaml"
    overrides = build_overrides(run, dry_run)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_config_simulation.py"),
        "--config",
        str(config_path),
    ] + overrides

    if verbose:
        print(f"\n→ {run['experiment_id']}")
        print("  " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"  FAILED: {run['experiment_id']}")
            if not verbose and result.stderr:
                print(result.stderr[-800:])
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {run['experiment_id']}")
        return False
    except Exception as exc:  # noqa: BLE001 — surface any runner failure
        print(f"  ERROR: {run['experiment_id']}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Placebo semantic-isolation ablation runner.")
    parser.add_argument("--dry-run", action="store_true", help="Use mock policy — fast, no GPU.")
    parser.add_argument("--agents", type=int, default=100, help="Population size (use ~6 for dry-run).")
    parser.add_argument("--rounds", type=int, default=30, help="Simulation rounds (use ~4 for dry-run).")
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated seeds.")
    parser.add_argument(
        "--arms",
        type=str,
        default="grounded,placebo,unconditioned",
        help="Comma-separated arms (default: all three).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"Unknown arm(s): {unknown}. Valid: {sorted(ARMS)}")

    runs = build_run_matrix(arms, seeds, args.agents, args.rounds)

    print(f"Placebo ablation: {len(runs)} runs planned")
    print(f"  Arms:    {arms}")
    print(f"  Seeds:   {seeds}")
    print(f"  Dry-run: {args.dry_run}")
    print()

    results = {"passed": [], "failed": []}
    for i, run in enumerate(runs, 1):
        print(f"[{i:2d}/{len(runs)}] {run['experiment_id']}", end=" ", flush=True)
        ok = run_single(run, dry_run=args.dry_run, verbose=args.verbose)
        print("OK" if ok else "FAIL")
        (results["passed"] if ok else results["failed"]).append(run["experiment_id"])

    print(f"\nDone: {len(results['passed'])} passed, {len(results['failed'])} failed")
    if results["failed"]:
        for r in results["failed"]:
            print(f"  FAILED: {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
