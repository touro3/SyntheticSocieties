"""
BGF Full Pipeline — One command does everything.

Stages:
  1. Run experiments (all policies × N seeds × R rounds)
  2. Generate comparison diagrams
  3. Run DuckDB analytics
  4. Print quick comparison table

Usage:
    # Full pipeline (baselines only, no GPU)
    python scripts/run_full_pipeline.py

    # Include LLM experiments (needs GPU)
    python scripts/run_full_pipeline.py --include-llm

    # Custom settings
    python scripts/run_full_pipeline.py --seeds 42,123,7 --rounds 10 --agents 20

    # Skip experiments, just regenerate plots from existing data
    python scripts/run_full_pipeline.py --plots-only

    # Include prompt perturbation experiments
    python scripts/run_full_pipeline.py --include-llm --include-perturbation
"""

import argparse
import json
import subprocess
import sys
import time
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import load_config
from utils.io import ensure_dir, save_yaml, set_global_seed
from scripts.run_config_simulation import run_simulation

POLICIES_BASELINE = ["template", "rule_based", "random"]
POLICIES_LLM = ["llm"]
PERTURBATION_MODES = ["rephrase", "shuffle", "noise"]

# Mapping from policy name → experiment ID prefix
POLICY_PREFIX = {
    "llm": "llm",
    "template": "template",
    "rule_based": "rule",
    "random": "random",
}


def parse_args():
    parser = argparse.ArgumentParser(description="BGF Full Pipeline")
    parser.add_argument("--seeds", type=str, default="42,123,7",
                        help="Seeds to use (comma-separated)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Simulation rounds per experiment")
    parser.add_argument("--agents", type=int, default=20,
                        help="Number of agents per experiment")
    parser.add_argument("--include-llm", action="store_true",
                        help="Include LLM policy experiments (needs GPU)")
    parser.add_argument("--include-perturbation", action="store_true",
                        help="Include prompt perturbation experiments")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip experiments, just regenerate plots")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip experiments that already have summary.json")
    parser.add_argument("--llm-ablation-level", type=int, default=5,
                        help="Prompt ablation level (0-5) for base LLM experiments")
    parser.add_argument("--run-ablation-ladder", action="store_true",
                        help="Run full V0-V5 ablation ladder suite")
    return parser.parse_args()


def experiment_exists(exp_id: str) -> bool:
    return (Path("experiments") / exp_id / "summary.json").exists()


def run_single_experiment(policy: str, seed: int, rounds: int, agents: int,
                          prefix: str, extra_overrides: list[str] = None) -> str:
    """Run a single experiment, return the experiment ID."""
    exp_prefix = POLICY_PREFIX.get(policy, policy)
    exp_id = f"{prefix}{exp_prefix}_s{seed}"

    # Build config overrides
    base_config = "configs/base_config.yaml"
    overrides = [
        f"project.experiment_id={exp_id}",
        f"project.seed={seed}",
        f"policy.type={policy}",
        f"simulation.rounds={rounds}",
        f"simulation.population_size={agents}",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)

    if policy == "llm" or "ablation_level" in str(extra_overrides):
        # Run in-process to retain the LLMBackend singleton across seeds!
        run_simulation(base_config, list(overrides))
    else:
        cmd = [
            sys.executable, "scripts/run_config_simulation.py",
            "--config", base_config,
        ] + list(overrides)
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    return exp_id



def run_experiments(args) -> list[str]:
    """Run all experiments and return list of experiment IDs."""
    if "," in args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        val = int(args.seeds)
        seeds = [42, 123, 7, 1, 2, 88, 99, 101, 102, 103][:val] if val <= 20 else [val]
    policies = POLICIES_BASELINE.copy()
    
    experiments = []
    # Regular baselines
    for policy in policies:
        for seed in seeds:
            experiments.append(("cmp_", policy, seed, []))

    if args.include_llm and not args.run_ablation_ladder:
        for seed in seeds:
            experiments.append(("cmp_", "llm", seed, [f"llm.ablation_level={args.llm_ablation_level}"]))

    # Full V0-V5 Sweep
    if args.run_ablation_ladder:
        for lvl in range(6):
            for seed in seeds:
                extra = [f"llm.ablation_level={lvl}"]
                if lvl >= 5:
                    extra.append("llm.temperature=0.7")
                experiments.append((f"abl_v{lvl}_", "llm", seed, extra))

    # Perturbation experiments
    if args.include_perturbation and (args.include_llm or args.run_ablation_ladder):
        for mode in PERTURBATION_MODES:
            for seed in seeds:
                experiments.append((
                    "pert_", "llm", seed,
                    [f"perturbation.mode={mode}", f"llm.ablation_level={args.llm_ablation_level}"]
                ))

    total = len(experiments)
    completed = 0
    skipped = 0
    exp_ids = []
    run_times = []  # track elapsed times for ETA

    n_llm = sum(1 for (_, p, _, _) in experiments if p == "llm")
    n_baseline = total - n_llm

    print(f"\n{'─' * 60}")
    print(f"  STAGE 1: Experiments ({total} total)")
    if n_llm > 0:
        print(f"  ⚠  {n_llm} LLM experiments (~5-8 min each, ~{n_llm * 6} min total)")
    print(f"{'─' * 60}")

    # Separate baseline (parallelizable) and LLM (sequential) experiments
    baseline_exps = [(i, p, pol, s, e) for i, (p, pol, s, e) in enumerate(experiments, 1)
                     if pol != "llm"]
    llm_exps = [(i, p, pol, s, e) for i, (p, pol, s, e) in enumerate(experiments, 1)
                if pol == "llm"]

    # Run baselines in parallel (no GPU needed)
    if baseline_exps:
        to_run = []
        for i, prefix, policy, seed, extra in baseline_exps:
            exp_prefix = POLICY_PREFIX.get(policy, policy)
            exp_id = f"{prefix}{exp_prefix}_s{seed}"
            if args.skip_existing and experiment_exists(exp_id):
                print(f"  [{i}/{total}] SKIP: {exp_id}")
                skipped += 1
                exp_ids.append(exp_id)
            else:
                to_run.append((i, prefix, policy, seed, extra, exp_id))

        if to_run:
            n_workers = min(len(to_run), 8)  # Up to 8 parallel baseline experiments
            print(f"  ⚡ Running {len(to_run)} baselines in parallel ({n_workers} workers)")
            t0_parallel = time.time()

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for i, prefix, policy, seed, extra, exp_id in to_run:
                    future = executor.submit(
                        run_single_experiment, policy, seed, args.rounds, args.agents, prefix, extra
                    )
                    futures[future] = (i, exp_id, policy)

                for future in as_completed(futures):
                    i, exp_id, policy = futures[future]
                    try:
                        future.result()
                        # Register sequentially to avoid race condition in tracker/experiment_index.parquet
                        subprocess.run([
                            sys.executable, "scripts/register_experiment.py",
                            "--run-dir", f"experiments/{exp_id}",
                        ], check=True, capture_output=True, text=True)
                        completed += 1
                        print(f"  [{i}/{total}] ✓ {exp_id}")
                    except Exception as e:
                        print(f"  [{i}/{total}] ✗ {exp_id}: {str(e)[:100]}")
                    exp_ids.append(exp_id)


            parallel_elapsed = time.time() - t0_parallel
            print(f"  ⚡ Baselines done in {parallel_elapsed:.1f}s (parallel)")

    # Run LLM experiments sequentially (GPU bound)
    for i, prefix, policy, seed, extra in llm_exps:
        exp_prefix = POLICY_PREFIX.get(policy, policy)
        exp_id = f"{prefix}{exp_prefix}_s{seed}"
        if extra:
            for o in extra:
                if "perturbation.mode=" in o:
                    mode = o.split("=")[1]
                    exp_id = f"pert_{mode}_s{seed}"
                
                # Use ablation prefix if doing the ladder sweep
                if "llm.ablation_level=" in o and prefix.startswith("abl_v"):
                    lvl = o.split("=")[1]
                    exp_id = f"abl_v{lvl}_{exp_prefix}_s{seed}"

        if args.skip_existing and experiment_exists(exp_id):
            print(f"  [{i}/{total}] SKIP: {exp_id}")
            skipped += 1
            exp_ids.append(exp_id)
            continue

        t0 = time.time()
        print(f"  [{i}/{total}] Running: {exp_id} (LLM)...", end="", flush=True)

        try:
            run_single_experiment(policy, seed, args.rounds, args.agents, prefix, extra)
            # Register sequentially
            subprocess.run([
                sys.executable, "scripts/register_experiment.py",
                "--run-dir", f"experiments/{exp_id}",
            ], check=True, capture_output=True, text=True)
            
            elapsed = time.time() - t0
            run_times.append(elapsed)
            completed += 1


            remaining_llm = sum(1 for (_, _, p, s, _) in llm_exps
                                if not experiment_exists(f"cmp_{POLICY_PREFIX.get(p, p)}_s{s}"))
            if remaining_llm > 0:
                avg = np.mean(run_times) if run_times else elapsed
                eta_min = remaining_llm * avg / 60
                print(f" ✓ ({elapsed:.1f}s) — ETA: ~{eta_min:.0f} min")
            else:
                print(f" ✓ ({elapsed:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(f" ✗ ERROR")
            print(f"    {e.stderr[:200] if e.stderr else 'No error output'}")

        exp_ids.append(exp_id)

    print(f"\n  Done: {completed} ran, {skipped} skipped, {total} total")
    return exp_ids


def run_plots(args):
    """Generate all comparison diagrams."""
    print(f"\n{'─' * 60}")
    print(f"  STAGE 2: Generating Diagrams")
    print(f"{'─' * 60}")

    seeds_arg = args.seeds if args.seeds else "42,123,7,1,2"
    subprocess.run([
        sys.executable, "scripts/plot_policy_comparison_full.py", "--seeds", seeds_arg
    ], check=True)

    # Publication analytics plots
    subprocess.run([
        sys.executable, "scripts/plot_all_analytics.py"
    ], check=True)

    # Advanced Trajectory Plots (Phase 13)
    seeds_arg = args.seeds if args.seeds else "42,123,7,1,2"
    subprocess.run([
        sys.executable, "scripts/plot_trajectories_full.py", "--seeds", seeds_arg
    ], check=True)


    # Also run the empirical analysis plots
    subprocess.run([
        sys.executable, "scripts/plot_empirical_analysis.py"
    ], check=True, capture_output=True, text=True)
    print("  ✓ All plot suites generated")


def run_analytics():
    """Run DuckDB analytics."""
    print(f"\n{'─' * 60}")
    print(f"  STAGE 3: DuckDB Analytics")
    print(f"{'─' * 60}")

    subprocess.run([
        sys.executable, "scripts/run_analytics.py"
    ], check=True)


def print_quick_comparison(seeds_str: str):
    """Print a quick comparison table from experiment summaries."""
    seeds = [int(s) for s in seeds_str.split(",")]
    policies = ["llm", "template", "rule_based", "random"]

    print(f"\n{'═' * 70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'═' * 70}")

    header = f"{'Policy':<25} {'Wealth μ':>10} {'Gini':>8} {'Coop%':>8} {'Work%':>8} {'Save%':>8}"
    print(header)
    print("─" * 70)

    for policy in policies:
        prefix = POLICY_PREFIX.get(policy, policy)
        all_wealth = []
        all_actions = Counter()

        for seed in seeds:
            exp_id = f"cmp_{prefix}_s{seed}"
            summary_path = Path("experiments") / exp_id / "summary.json"
            if not summary_path.exists():
                continue

            with summary_path.open() as f:
                summary = json.loads(f.read())

            wealth_vals = summary.get("wealth", {}).get("values", [])
            all_wealth.extend(wealth_vals)

            eac = summary.get("event_action_counts", {})
            all_actions.update(eac)

        if not all_wealth:
            print(f"  {policy:<23} {'(no data)':>10}")
            continue

        mean_w = np.mean(all_wealth)
        # Gini
        arr = np.array(all_wealth)
        n = len(arr)
        if n > 1 and arr.sum() > 0:
            diff_sum = sum(abs(arr[i] - arr[j]) for i in range(n) for j in range(n))
            gini = diff_sum / (2 * n * n * arr.mean())
        else:
            gini = 0.0

        total_a = max(sum(all_actions.values()), 1)
        coop = all_actions.get("cooperate", 0) / total_a * 100
        work = all_actions.get("work", 0) / total_a * 100
        save = all_actions.get("save", 0) / total_a * 100

        label = {"llm": "LLM (Mistral-7B)", "template": "Template (ESS)",
                 "rule_based": "Rule-Based", "random": "Random"}[policy]

        print(f"  {label:<23} {mean_w:>10.1f} {gini:>8.3f} {coop:>7.1f}% {work:>7.1f}% {save:>7.1f}%")

    print(f"{'═' * 70}")

    # List available figures
    fig_dir = Path("analysis/figures")
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))
        print(f"\n  📊 Figures ({len(figs)}):")
        for f in figs:
            print(f"     {f}")

    # List analytics tables
    table_dir = Path("analysis/tables")
    if table_dir.exists():
        tables = sorted(table_dir.glob("*.csv"))
        if tables:
            print(f"\n  📋 Analytics Tables ({len(tables)}):")
            for t in tables:
                print(f"     {t}")

    print()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        BGF Full Pipeline — Synthetic Societies          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Rounds:   {args.rounds}")
    print(f"  Agents:   {args.agents}")
    print(f"  LLM:      {'Yes' if args.include_llm else 'No (add --include-llm)'}")
    print(f"  Perturb:  {'Yes' if args.include_perturbation else 'No'}")

    t0 = time.time()

    if not args.plots_only:
        run_experiments(args)
    else:
        print("\n  Skipping experiments (--plots-only)")

    run_plots(args)

    run_analytics()
    print_quick_comparison(args.seeds)

    elapsed = time.time() - t0
    print(f"  ⏱  Total pipeline time: {elapsed:.1f}s")
    print("  ✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()
