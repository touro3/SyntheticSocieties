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
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Must be set before any CUDA tensor is allocated (including in subprocesses
# that inherit this environment) so PyTorch's expandable-segments allocator
# is active from the start.  This prevents fragmentation-OOM on multi-seed
# runs where the singleton LLM model (14.5 GB) stays resident across seeds.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_config_simulation import run_simulation

POLICIES_BASELINE = ["template", "rule_based", "random", "rule_based_ess"]
POLICIES_LLM = ["llm"]
PERTURBATION_MODES = ["rephrase", "shuffle", "noise"]

# Mapping from policy name → experiment ID prefix
POLICY_PREFIX = {
    "llm": "llm",
    "template": "template",
    "rule_based": "rule",
    "random": "random",
    "rule_based_ess": "rbe",
}


def parse_seed_list(seeds_arg):
    """
    Parses a seed string into a list of integers.
    Supports single ints ('42'), comma-separated ('42,123,7'),
    and inclusive ranges ('1..10').
    """
    seeds = []
    # Split by comma to handle standard lists
    for part in seeds_arg.split(","):
        part = part.strip()
        if ".." in part:
            # Handle range syntax (e.g., '1..10')
            start_str, end_str = part.split("..")
            start = int(start_str)
            end = int(end_str)
            # Use end + 1 to make the range inclusive
            seeds.extend(range(start, end + 1))
        else:
            # Handle single integers
            seeds.append(int(part))

    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description="BGF Full Pipeline")
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Seeds to use (comma-separated)")
    parser.add_argument("--rounds", type=int, default=10, help="Simulation rounds per experiment")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents per experiment")
    parser.add_argument("--include-llm", action="store_true", help="Include LLM policy experiments (needs GPU)")
    parser.add_argument("--include-perturbation", action="store_true", help="Include prompt perturbation experiments")
    parser.add_argument("--plots-only", action="store_true", help="Skip experiments, just regenerate plots")
    parser.add_argument(
        "--skip-existing", action="store_true", default=False, help="Skip experiments that already have summary.json"
    )
    parser.add_argument(
        "--llm-ablation-level", type=int, default=5, help="Prompt ablation level (0-5) for base LLM experiments"
    )
    parser.add_argument("--run-ablation-ladder", action="store_true", help="Run full V0-V5 ablation ladder suite")
    parser.add_argument(
        "--analytics-scope",
        choices=["run", "global"],
        default="run",
        help="Analytics scope: current run only (default) or full historical index.",
    )
    parser.add_argument(
        "--analytics-output-dir",
        type=str,
        default="analysis/tables",
        help="Output directory for analytics CSV tables.",
    )
    parser.add_argument(
        "--skip-integrity-audit",
        action="store_true",
        help="Skip post-run research integrity audit.",
    )
    parser.add_argument(
        "--integrity-level",
        choices=["basic", "publication"],
        default="basic",
        help="Research integrity audit strictness.",
    )
    parser.add_argument(
        "--condition",
        choices=["A", "B", "C", "D"],
        default=None,
        help=(
            "Experimental condition: A=baseline (no grounding), "
            "B=full grounding (default), C=counterfactual identity (soul swap), "
            "D=rule-based ESS only (no LLM, deterministic)."
        ),
    )
    return parser.parse_args()


def experiment_exists(exp_id: str) -> bool:
    return (Path("experiments") / exp_id / "summary.json").exists()


def run_single_experiment(
    policy: str, seed: int, rounds: int, agents: int, prefix: str, extra_overrides: list[str] = None
) -> str:
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
            sys.executable,
            "scripts/run_config_simulation.py",
            "--config",
            base_config,
        ] + list(overrides)
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    return exp_id


def run_experiments(args) -> list[str]:
    """Run all experiments and return list of experiment IDs."""
    seeds = parse_seed_list(args.seeds)
    policies = POLICIES_BASELINE.copy()

    # ── Condition-based overrides ─────────────────────────────────────────
    condition_overrides: list[str] = []
    if getattr(args, "condition", None) == "A":
        condition_overrides.append("llm.ablation_level=0")
    elif getattr(args, "condition", None) == "C":
        condition_overrides.append("agent_defaults.shuffle_traits=true")
    elif getattr(args, "condition", None) == "D":
        # Condition D: deterministic rule-based ESS policy only, no LLM needed
        policies = ["rule_based_ess"]

    experiments = []
    # Regular baselines
    for policy in policies:
        for seed in seeds:
            experiments.append(("cmp_", policy, seed, list(condition_overrides)))

    if args.include_llm and not args.run_ablation_ladder:
        for seed in seeds:
            experiments.append(
                ("cmp_", "llm", seed, [f"llm.ablation_level={args.llm_ablation_level}"] + condition_overrides)
            )

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
                experiments.append(
                    (
                        "pert_",
                        "llm",
                        seed,
                        [f"perturbation.mode={mode}", f"llm.ablation_level={args.llm_ablation_level}"],
                    )
                )

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
    baseline_exps = [(i, p, pol, s, e) for i, (p, pol, s, e) in enumerate(experiments, 1) if pol != "llm"]
    llm_exps = [(i, p, pol, s, e) for i, (p, pol, s, e) in enumerate(experiments, 1) if pol == "llm"]

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
                        subprocess.run(
                            [
                                sys.executable,
                                "scripts/register_experiment.py",
                                "--run-dir",
                                f"experiments/{exp_id}",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        completed += 1
                        print(f"  [{i}/{total}] ✓ {exp_id}")
                    except Exception as e:
                        print(f"  [{i}/{total}] ✗ {exp_id}: {str(e)[:100]}")
                    exp_ids.append(exp_id)

            parallel_elapsed = time.time() - t0_parallel
            print(f"  ⚡ Baselines done in {parallel_elapsed:.1f}s (parallel)")

    # Run LLM experiments sequentially (GPU bound)
    _first_llm = True
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

        # Release KV-cache fragments from the previous seed before starting
        # the next one.  The model stays loaded (no reload cost).
        if not _first_llm:
            from decision.llm_backend import LLMBackend

            LLMBackend.between_seeds()
        _first_llm = False

        t0 = time.time()
        print(f"  [{i}/{total}] Running: {exp_id} (LLM)...", end="", flush=True)

        try:
            run_single_experiment(policy, seed, args.rounds, args.agents, prefix, extra)
            # Register sequentially
            subprocess.run(
                [
                    sys.executable,
                    "scripts/register_experiment.py",
                    "--run-dir",
                    f"experiments/{exp_id}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            elapsed = time.time() - t0
            run_times.append(elapsed)
            completed += 1

            remaining_llm = sum(
                1 for (_, _, p, s, _) in llm_exps if not experiment_exists(f"cmp_{POLICY_PREFIX.get(p, p)}_s{s}")
            )
            if remaining_llm > 0:
                avg = np.mean(run_times) if run_times else elapsed
                eta_min = remaining_llm * avg / 60
                print(f" ✓ ({elapsed:.1f}s) — ETA: ~{eta_min:.0f} min")
            else:
                print(f" ✓ ({elapsed:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(" ✗ ERROR")
            print(f"    {e.stderr[:200] if e.stderr else 'No error output'}")

        exp_ids.append(exp_id)

    print(f"\n  Done: {completed} ran, {skipped} skipped, {total} total")
    return exp_ids


def run_plots(args):
    """Generate all comparison diagrams."""
    print(f"\n{'─' * 60}")
    print("  STAGE 2: Generating Diagrams")
    print(f"{'─' * 60}")

    seeds_arg = args.seeds if args.seeds else "42,123,7,1,2"
    subprocess.run([sys.executable, "scripts/plot_policy_comparison_full.py", "--seeds", seeds_arg], check=True)

    # Publication analytics plots
    subprocess.run([sys.executable, "scripts/plot_all_analytics.py"] + ["--seeds", args.seeds], check=True)

    # Advanced Trajectory Plots (Phase 13)
    seeds_arg = args.seeds if args.seeds else "42,123,7,1,2"
    subprocess.run([sys.executable, "scripts/plot_trajectories_full.py", "--seeds", seeds_arg], check=True)

    # Also run the empirical analysis plots
    subprocess.run([sys.executable, "scripts/plot_empirical_analysis.py"], check=True, capture_output=True, text=True)
    print("  ✓ All plot suites generated")


def _infer_cmp_ids(seeds: list[int], include_llm: bool) -> list[str]:
    cmp_ids = []
    for s in seeds:
        cmp_ids.extend(
            [
                f"cmp_template_s{s}",
                f"cmp_rule_s{s}",
                f"cmp_random_s{s}",
            ]
        )
        if include_llm:
            cmp_ids.append(f"cmp_llm_s{s}")
    return cmp_ids


def run_analytics(args, exp_ids: list[str]):
    """Run DuckDB analytics."""
    print(f"\n{'─' * 60}")
    print("  STAGE 3: DuckDB Analytics")
    print(f"{'─' * 60}")

    cmd = [
        sys.executable,
        "scripts/run_analytics.py",
        "--output-dir",
        args.analytics_output_dir,
    ]

    if args.analytics_scope == "run":
        seeds = parse_seed_list(args.seeds)
        policies = ["template", "rule_based", "random"] + (["llm"] if args.include_llm else [])
        cmp_ids = (
            sorted({x for x in exp_ids if x.startswith("cmp_")}) if exp_ids else _infer_cmp_ids(seeds, args.include_llm)
        )
        if cmp_ids:
            cmd += ["--experiment-ids", ",".join(cmp_ids)]
        cmd += ["--seeds", ",".join(str(s) for s in seeds)]
        cmd += ["--policy-types", ",".join(policies), "--require-cmp-only"]

    subprocess.run(cmd, check=True)


def run_integrity_audit(args, manifest_path: Path) -> None:
    """Run post-pipeline research integrity checks."""
    print(f"\n{'─' * 60}")
    print("  STAGE 4: Research Integrity Audit")
    print(f"{'─' * 60}")
    cmd = [
        sys.executable,
        "scripts/research_integrity_audit.py",
        "--manifest",
        str(manifest_path),
        "--llm-vs-baselines",
        str(Path(args.analytics_output_dir) / "llm_vs_baselines.csv"),
        "--policy-comparison",
        str(Path(args.analytics_output_dir) / "policy_comparison.csv"),
        "--cross-cultural",
        "analysis/cross_cultural_expanded_results.json",
        "--human-baseline",
        "analysis/tables/human_baseline_metrics.json",
        "--level",
        args.integrity_level,
        "--output-json",
        "analysis/reports/research_integrity_audit.json",
        "--output-markdown",
        "analysis/reports/research_integrity_audit.md",
    ]
    if args.include_llm:
        cmd.append("--fail-on-blockers")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("  ✓ Integrity audit passed")
    else:
        print("  ⚠ Integrity audit found issues (see analysis/reports/research_integrity_audit.md)")


def write_run_manifest(start_ts: float, args, exp_ids: list[str]) -> Path:
    """Write an auditable manifest of the current pipeline run."""
    reports_dir = Path("analysis/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_dir / "last_pipeline_run_manifest.json"

    def _modified_since(base: Path, patterns: list[str], since: float) -> list[str]:
        out = []
        for pattern in patterns:
            for p in base.glob(pattern):
                try:
                    if p.is_file() and p.stat().st_mtime >= since:
                        out.append(str(p))
                except FileNotFoundError:
                    continue
        return sorted(set(out))

    seeds = parse_seed_list(args.seeds)
    summary_paths = []
    for exp_id in exp_ids:
        p = Path("experiments") / exp_id / "summary.json"
        if p.exists():
            summary_paths.append(str(p))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "seeds": seeds,
            "rounds": args.rounds,
            "agents": args.agents,
            "include_llm": args.include_llm,
            "include_perturbation": args.include_perturbation,
            "plots_only": args.plots_only,
            "analytics_scope": args.analytics_scope,
            "analytics_output_dir": args.analytics_output_dir,
            "integrity_level": args.integrity_level,
            "skip_integrity_audit": args.skip_integrity_audit,
        },
        "experiment_ids": exp_ids,
        "summary_paths": summary_paths,
        "analysis_outputs_modified_since_run_start": {
            "figures": _modified_since(Path("analysis/figures"), ["*.png", "*.pdf"], start_ts),
            "tables": _modified_since(Path(args.analytics_output_dir), ["*.csv", "*.json"], start_ts),
        },
    }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def print_quick_comparison(seeds_str: str):
    """Print a quick comparison table from experiment summaries."""
    seeds = parse_seed_list(seeds_str)
    policies = ["llm", "template", "rule_based", "random"]

    print(f"\n{'═' * 70}")
    print("  RESULTS COMPARISON")
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

        label = {
            "llm": "LLM (Mistral-7B)",
            "template": "Template (ESS)",
            "rule_based": "Rule-Based",
            "random": "Random",
            "rule_based_ess": "Rule-Based ESS (D)",
        }.get(policy, policy)

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
    exp_ids: list[str] = []

    if not args.plots_only:
        exp_ids = run_experiments(args)
    else:
        print("\n  Skipping experiments (--plots-only)")
        exp_ids = _infer_cmp_ids(parse_seed_list(args.seeds), args.include_llm)

    run_plots(args)

    run_analytics(args, exp_ids)
    print_quick_comparison(args.seeds)

    manifest_path = write_run_manifest(t0, args, exp_ids)
    print(f"  🧾 Run manifest: {manifest_path}")

    if not args.skip_integrity_audit:
        run_integrity_audit(args, manifest_path)
    else:
        print("  ⏭  Integrity audit skipped (--skip-integrity-audit)")

    elapsed = time.time() - t0
    print(f"  ⏱  Total pipeline time: {elapsed:.1f}s")
    print("  ✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()
