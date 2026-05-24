"""Config-driven experiment matrix runner.

Runs a full Cartesian grid of (seeds × conditions × hyperparameter sweeps)
and emits per-cell summary JSON plus a combined results CSV.

Usage:
    # Dry run — print the matrix without executing
    python scripts/run_experiment_matrix.py --dry-run

    # Run A/B comparison, 5 seeds, 30 rounds, 100 agents
    python scripts/run_experiment_matrix.py \\
        --conditions A B \\
        --seeds 1..5 \\
        --rounds 30 --agents 100

    # Full ablation sweep (V0-V5) across 3 seeds
    python scripts/run_experiment_matrix.py \\
        --conditions B \\
        --ablation-levels 0 1 2 3 4 5 \\
        --seeds 1,2,3

    # Temperature sweep
    python scripts/run_experiment_matrix.py \\
        --conditions B \\
        --temperatures 0.3 0.5 0.7 1.0 \\
        --seeds 1,2,3

    # Include LLM (GPU required)
    python scripts/run_experiment_matrix.py --include-llm \\
        --conditions A B C D --seeds 1..10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Condition definitions ─────────────────────────────────────────────────────

_CONDITION_OVERRIDES: dict[str, list[str]] = {
    # population.source=empirical is critical for both A and B: agents must have
    # realistic ESS-sampled demographic profiles. Condition A = no prompt grounding
    # (ablation=0); Condition B = full ESS context injected at decision time (ablation=5).
    "A": ["population.source=empirical", "llm.ablation_level=0"],  # Pure LLM, no grounding
    "B": ["population.source=empirical", "llm.ablation_level=5"],  # Full grounding
    "C": ["policy.type=generative_agents"],  # Generative Agents proxy
    "D": ["policy.type=rule_based_ess"],  # Deterministic ESS rule-based
}

_CONDITION_POLICY: dict[str, str] = {
    "A": "llm",
    "B": "llm",
    "C": "generative_agents",
    "D": "rule_based_ess",
}


# ── Seed parsing (shared with run_full_pipeline) ──────────────────────────────


def parse_seed_list(seeds_str: str) -> list[int]:
    seeds: list[int] = []
    for part in seeds_str.split(","):
        part = part.strip()
        if ".." in part:
            lo, hi = part.split("..")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


# ── Matrix cell ───────────────────────────────────────────────────────────────


def _cell_id(condition: str, seed: int, ablation: int | None, temperature: float | None, suffix: str = "") -> str:
    parts = [f"mx_{condition}{('_' + suffix) if suffix else ''}_s{seed}"]
    if ablation is not None:
        parts.append(f"abl{ablation}")
    if temperature is not None:
        parts.append(f"t{str(temperature).replace('.', 'p')}")
    return "_".join(parts)


def _build_overrides(
    condition: str,
    seed: int,
    rounds: int,
    agents: int,
    ablation: int | None,
    temperature: float | None,
    exp_id: str,
) -> list[str]:
    overrides = [
        f"project.experiment_id={exp_id}",
        f"project.seed={seed}",
        f"policy.type={_CONDITION_POLICY[condition]}",
        f"simulation.rounds={rounds}",
        f"simulation.population_size={agents}",
    ]
    overrides.extend(_CONDITION_OVERRIDES.get(condition, []))
    if ablation is not None:
        overrides.append(f"llm.ablation_level={ablation}")
    if temperature is not None:
        overrides.append(f"llm.temperature={temperature}")
    return overrides


# ── Matrix execution ──────────────────────────────────────────────────────────


def run_matrix(args) -> list[dict]:
    """Execute the experiment grid and return per-cell result dicts."""
    from scripts.run_config_simulation import run_simulation

    seeds = parse_seed_list(args.seeds)
    ablation_levels: list[int | None] = [None]
    temperatures: list[float | None] = [None]

    if args.ablation_levels:
        ablation_levels = [int(x) for x in args.ablation_levels]
    if args.temperatures:
        temperatures = [float(x) for x in args.temperatures]

    # Build the full cell list
    cells = list(product(args.conditions, seeds, ablation_levels, temperatures))
    total = len(cells)

    print(f"\n{'─' * 64}")
    print(f"  Experiment Matrix: {total} cells")
    print(f"  Conditions: {args.conditions}  Seeds: {seeds}")
    if args.ablation_levels:
        print(f"  Ablation levels: {ablation_levels}")
    if args.temperatures:
        print(f"  Temperatures: {temperatures}")
    print(f"{'─' * 64}")

    suffix = getattr(args, "id_suffix", "") or ""

    if args.dry_run:
        print("\nDRY RUN — cells that would be executed:\n")
        for i, (cond, seed, abl, temp) in enumerate(cells, 1):
            eid = _cell_id(cond, seed, abl, temp, suffix)
            print(f"  {i:3d}. {eid}")
        print(f"\nTotal: {total} cells")
        return []

    results: list[dict] = []
    completed = skipped = failed = 0

    for i, (cond, seed, abl, temp) in enumerate(cells, 1):
        exp_id = _cell_id(cond, seed, abl, temp, suffix)
        summary_path = Path("experiments") / exp_id / "summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"  [{i:3d}/{total}] SKIP  {exp_id}")
            skipped += 1
            results.append(_load_summary(exp_id, cond, seed, abl, temp))
            continue

        # Release GPU KV-cache between LLM seeds
        if cond in ("A", "B") and i > 1:
            try:
                from decision.llm_backend import LLMBackend

                LLMBackend.between_seeds()
            except ImportError:
                pass

        overrides = _build_overrides(cond, seed, args.rounds, args.agents, abl, temp, exp_id)

        t0 = time.time()
        print(f"  [{i:3d}/{total}] RUN   {exp_id} ...", end="", flush=True)

        try:
            run_simulation("configs/base_config.yaml", overrides)
            elapsed = time.time() - t0
            print(f" ✓ ({elapsed:.1f}s)")
            completed += 1
            results.append(_load_summary(exp_id, cond, seed, abl, temp))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f" ✗ ({elapsed:.1f}s): {str(exc)[:80]}")
            failed += 1
            results.append(
                {
                    "exp_id": exp_id,
                    "condition": cond,
                    "seed": seed,
                    "ablation": abl,
                    "temperature": temp,
                    "status": "failed",
                    "error": str(exc)[:200],
                }
            )

    print(f"\n  Done: {completed} ran, {skipped} skipped, {failed} failed  ({total} total)")
    return results


def _load_summary(exp_id: str, cond: str, seed: int, abl, temp) -> dict:
    path = Path("experiments") / exp_id / "summary.json"
    row: dict = {
        "exp_id": exp_id,
        "condition": cond,
        "seed": seed,
        "ablation": abl,
        "temperature": temp,
        "status": "ok",
    }
    if path.exists():
        try:
            data = json.loads(path.read_text())
            metrics = data.get("metrics", {}) or {}

            def _first(*keys):
                for k in keys:
                    src = metrics
                    for part in k.split("."):
                        if not isinstance(src, dict) or part not in src:
                            src = None
                            break
                        src = src[part]
                    if src is not None:
                        return src
                # Fall back to data root using same dotted-path lookup
                for k in keys:
                    src = data
                    for part in k.split("."):
                        if not isinstance(src, dict) or part not in src:
                            src = None
                            break
                        src = src[part]
                    if src is not None:
                        return src
                return None

            wealth_vals = _first("wealth.values") or []
            row["cooperation_rate"] = _first("cooperation_rate", "behavior.cooperation_rate")
            row["gini"] = _first("gini", "wealth.gini")
            if row["gini"] is None and wealth_vals:
                row["gini"] = _compute_gini(wealth_vals)
            row["mean_wealth"] = _first("mean_wealth", "wealth.mean")
            if row["mean_wealth"] is None and wealth_vals:
                row["mean_wealth"] = sum(wealth_vals) / len(wealth_vals)
            row["brm"] = _first("brm", "brm_composite", "behavioral_realism.brm")
            row["fidelity"] = _first("persona_fidelity", "fidelity.persona_fidelity")
        except Exception:
            row["status"] = "parse_error"
    else:
        row["status"] = "missing"
    return row


def _compute_gini(values) -> float:
    v = sorted(float(x) for x in values if x is not None)
    n = len(v)
    if n == 0:
        return 0.0
    s = sum(v)
    if s <= 0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(v, 1))
    return (2 * cum) / (n * s) - (n + 1) / n


# ── Statistical summary ───────────────────────────────────────────────────────


def _print_stats(results: list[dict]) -> None:
    """Print per-condition mean ± std for key metrics."""
    try:
        import numpy as np

        from metrics.statistical_inference import power_report
    except ImportError:
        return

    conditions = sorted({r["condition"] for r in results if r.get("status") == "ok"})
    metrics = ["cooperation_rate", "gini", "mean_wealth", "brm"]

    print(f"\n{'═' * 72}")
    print("  MATRIX RESULTS (mean ± std, 95% CI)")
    print(f"{'═' * 72}")

    cond_data: dict[str, dict[str, list]] = {}
    for row in results:
        if row.get("status") != "ok":
            continue
        cond = row["condition"]
        if cond not in cond_data:
            cond_data[cond] = {m: [] for m in metrics}
        for m in metrics:
            v = row.get(m)
            if v is not None:
                cond_data[cond][m].append(float(v))

    for m in metrics:
        print(f"\n  {m}:")
        for cond in conditions:
            vals = cond_data.get(cond, {}).get(m, [])
            if not vals:
                print(f"    {cond}: (no data)")
                continue
            arr = np.array(vals)
            print(f"    {cond}: {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")

    # A vs B power report if both present
    if "A" in cond_data and "B" in cond_data:
        for m in ["cooperation_rate", "gini"]:
            a_vals = cond_data["A"].get(m, [])
            b_vals = cond_data["B"].get(m, [])
            if len(a_vals) >= 3 and len(b_vals) >= 3:
                try:
                    rpt = power_report(a_vals, b_vals)
                    print(
                        f"\n  A vs B [{m}]: d={rpt['cohens_d']:.3f} ({rpt['interpretation']}), "
                        f"p={rpt['mann_whitney']['p_value']:.4f} "
                        f"({'*' if rpt['mann_whitney']['significant'] else 'ns'}), "
                        f"min_seeds_80%={rpt['min_seeds_80pct_power']}"
                    )
                except Exception:
                    pass

    print(f"\n{'═' * 72}")


# ── CSV export ────────────────────────────────────────────────────────────────


def _save_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp_id",
        "condition",
        "seed",
        "ablation",
        "temperature",
        "status",
        "cooperation_rate",
        "gini",
        "mean_wealth",
        "brm",
        "fidelity",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results CSV: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="BGF Config-Driven Experiment Matrix")
    p.add_argument(
        "--conditions", nargs="+", default=["A", "B"], choices=["A", "B", "C", "D"], help="Conditions to run"
    )
    p.add_argument("--seeds", type=str, default="1,2,3", help="Seeds: '1,2,3' or '1..10'")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--agents", type=int, default=100)
    p.add_argument(
        "--ablation-levels",
        nargs="*",
        type=int,
        default=None,
        help="Ablation levels (0-5); overrides condition defaults",
    )
    p.add_argument("--temperatures", nargs="*", type=float, default=None, help="Temperature sweep values")
    p.add_argument("--include-llm", action="store_true", help="Include LLM conditions (requires GPU)")
    p.add_argument("--skip-existing", action="store_true", help="Skip cells with existing summary.json")
    p.add_argument("--dry-run", action="store_true", help="Print matrix without running")
    p.add_argument("--output-csv", type=str, default="analysis/tables/experiment_matrix_results.csv")
    p.add_argument("--id-suffix", type=str, default="", help="Inserted after the condition in cell IDs (e.g. 'n500' → mx_A_n500_s1) to disambiguate from prior matrix runs at a different scale")
    return p.parse_args()


def main():
    args = parse_args()

    # Non-LLM conditions only unless --include-llm
    if not args.include_llm:
        args.conditions = [c for c in args.conditions if c == "D"]
        if not args.conditions:
            args.conditions = ["D"]
            print("  Note: --include-llm not set; only Condition D (rule-based) will run.")
            print("  Add --include-llm to include LLM conditions A/B/C.")

    results = run_matrix(args)

    if results:
        _print_stats(results)
        _save_csv(results, Path(args.output_csv))


if __name__ == "__main__":
    main()
