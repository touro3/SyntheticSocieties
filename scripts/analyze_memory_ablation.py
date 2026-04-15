"""
Memory Ablation Analysis (Phase 29.4).

Loads completed ablation experiment directories, computes per-condition
metrics, and produces the 2×4 factorial table + interaction plot.

Usage:
    # Real data:
    python scripts/analyze_memory_ablation.py --exp-dir experiments/

    # Dry-run with synthetic data (no experiments needed):
    python scripts/analyze_memory_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MEMORY_LEVELS = [0, 1, 2, 3]
LEVEL_LABELS = {0: "M0 (none)", 1: "M1 (window)", 2: "M2 (archive)", 3: "M3 (full)"}
CONDITIONS = ["grounded", "ungrounded"]
SEEDS = [42, 123, 7]

# Metrics extracted per run
METRIC_KEYS = [
    "cooperation_rate",
    "wealth_gini",
    "wealth_mean",
    "persona_fidelity",
]


def _exp_id(level: int, condition: str, seed: int) -> str:
    names = {0: "M0_no_memory", 1: "M1_window", 2: "M2_archive", 3: "M3_full"}
    return f"ablation_{names[level]}_{condition}_s{seed}"


def load_run_metrics(exp_dir: Path, exp_id: str) -> dict | None:
    """Load metrics from an experiment directory. Returns None if missing."""
    run_dir = exp_dir / exp_id
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"

    metrics: dict = {}

    if metrics_path.exists():
        try:
            metrics.update(json.loads(metrics_path.read_text()))
        except Exception:
            pass

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            # Extract cooperation rate from action distribution
            action_dist = summary.get("action_distribution", {})
            total_actions = sum(action_dist.values()) or 1
            metrics["cooperation_rate"] = action_dist.get("cooperate", 0) / total_actions
            metrics["wealth_mean"] = summary.get("wealth_mean", metrics.get("wealth_mean"))
            metrics["wealth_gini"] = summary.get("wealth_gini", metrics.get("wealth_gini"))
        except Exception:
            pass

    if not metrics:
        return None
    return metrics


def generate_synthetic_metrics(level: int, condition: str, seed: int) -> dict:
    """Generate plausible synthetic metrics for dry-run validation."""
    rng = random.Random(seed + level * 100 + (0 if condition == "grounded" else 500))

    # Grounded + higher level → higher cooperation + lower gini
    base_coop = 0.25 + 0.05 * level + (0.08 if condition == "grounded" else 0.0)
    base_gini = 0.40 - 0.02 * level - (0.05 if condition == "grounded" else 0.0)
    base_fidelity = 0.50 + 0.06 * level + (0.10 if condition == "grounded" else 0.0)

    return {
        "cooperation_rate": max(0.0, min(1.0, base_coop + rng.gauss(0, 0.03))),
        "wealth_gini": max(0.0, min(1.0, base_gini + rng.gauss(0, 0.02))),
        "wealth_mean": 50.0 + rng.gauss(0, 5.0),
        "persona_fidelity": max(0.0, min(1.0, base_fidelity + rng.gauss(0, 0.04))),
    }


def build_table(data: dict[tuple, list[dict]]) -> dict:
    """Aggregate per-seed metrics into mean ± std for each cell."""
    table = {}
    for (level, cond), runs in data.items():
        if not runs:
            continue
        cell = {}
        for key in METRIC_KEYS:
            values = [r[key] for r in runs if r.get(key) is not None]
            if values:
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
                cell[key] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}
        table[(level, cond)] = cell
    return table


def print_table(table: dict) -> None:
    """Print a readable 2×4 factorial table."""
    header = f"{'Level':<20}" + "".join(f"  {c:<12}" for c in CONDITIONS)
    print("\n=== Memory Ablation: Cooperation Rate (mean ± std) ===")
    print(header)
    print("-" * len(header))
    for level in MEMORY_LEVELS:
        row = f"{LEVEL_LABELS[level]:<20}"
        for cond in CONDITIONS:
            cell = table.get((level, cond), {}).get("cooperation_rate")
            if cell:
                row += f"  {cell['mean']:.3f}±{cell['std']:.3f}  "
            else:
                row += "  N/A           "
        print(row)

    print("\n=== Memory Ablation: Persona Fidelity (mean ± std) ===")
    print(header)
    print("-" * len(header))
    for level in MEMORY_LEVELS:
        row = f"{LEVEL_LABELS[level]:<20}"
        for cond in CONDITIONS:
            cell = table.get((level, cond), {}).get("persona_fidelity")
            if cell:
                row += f"  {cell['mean']:.3f}±{cell['std']:.3f}  "
            else:
                row += "  N/A           "
        print(row)


def save_results(table: dict, out_path: Path) -> None:
    """Serialize table to JSON, converting tuple keys to strings."""
    serializable = {f"level{level}_{cond}": cells for (level, cond), cells in table.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nTable saved to: {out_path}")


def plot_interaction(table: dict, out_path: Path) -> None:
    """Generate cooperation rate interaction plot (requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping interaction plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_to_plot = [("cooperation_rate", "Cooperation Rate"), ("persona_fidelity", "Persona Fidelity")]

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        for cond in CONDITIONS:
            xs, ys, yerrs = [], [], []
            for level in MEMORY_LEVELS:
                cell = table.get((level, cond), {}).get(metric)
                if cell:
                    xs.append(level)
                    ys.append(cell["mean"])
                    yerrs.append(cell["std"])
            if xs:
                ax.errorbar(xs, ys, yerr=yerrs, marker="o", label=cond, capsize=4)
        ax.set_xlabel("Memory Level")
        ax.set_ylabel(label)
        ax.set_title(f"Memory Ablation: {label}")
        ax.set_xticks(MEMORY_LEVELS)
        ax.set_xticklabels([LEVEL_LABELS[l] for l in MEMORY_LEVELS], rotation=15)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Interaction plot saved to: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory ablation analysis.")
    parser.add_argument("--exp-dir", type=str, default="experiments/", help="Path to experiments directory.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data instead of real experiments.")
    parser.add_argument(
        "--out", type=str, default="analysis/tables/memory_ablation.json", help="Output JSON path for the result table."
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="analysis/figures/memory_ablation_interaction.png",
        help="Output PNG path for the interaction plot.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    data: dict[tuple, list[dict]] = {}

    missing = 0
    found = 0
    for level in MEMORY_LEVELS:
        for cond in CONDITIONS:
            data[(level, cond)] = []
            for seed in SEEDS:
                exp_id = _exp_id(level, cond, seed)
                if args.dry_run:
                    row = generate_synthetic_metrics(level, cond, seed)
                    data[(level, cond)].append(row)
                    found += 1
                else:
                    row = load_run_metrics(exp_dir, exp_id)
                    if row is not None:
                        data[(level, cond)].append(row)
                        found += 1
                    else:
                        missing += 1
                        print(f"  Missing: {exp_id}")

    print(f"\nLoaded {found} runs, {missing} missing.")

    table = build_table(data)
    print_table(table)
    save_results(table, Path(args.out))
    plot_interaction(table, Path(args.plot))


if __name__ == "__main__":
    main()
