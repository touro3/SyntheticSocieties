import argparse
from pathlib import Path
import sys
    
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np

# Import the metrics calculator you just built
from metrics.macro_metrics import SocietyMacroMetrics

def main():
    parser = argparse.ArgumentParser(description="Plot Phase C Comparison")
    parser.add_argument("--data-dir", type=str, default="experiments/phase_c_comparison", help="Directory with parquet logs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    file_a = data_dir / "condition_a_events.parquet"
    file_b = data_dir / "condition_b_events.parquet"

    if not file_a.exists() or not file_b.exists():
        print(f"Error: Parquet logs not found in {data_dir}. Run the simulation first.")
        return

    print("Calculating macro trajectories...")
    # Analyze both conditions
    trends_a = SocietyMacroMetrics.analyze_trajectory(str(file_a)).to_pandas()
    trends_b = SocietyMacroMetrics.analyze_trajectory(str(file_b)).to_pandas()

    # Setup Plotting Style
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Wealth Inequality (Gini) ---
    axes[0].plot(trends_a["round_id"], trends_a["gini_coefficient"], label="Condition A (Ablated LLM)", color="crimson", linewidth=2.5, linestyle="--")
    axes[0].plot(trends_b["round_id"], trends_b["gini_coefficient"], label="Condition B (Grounded ESS LLM)", color="royalblue", linewidth=2.5)
    
    axes[0].set_title("Wealth Inequality Dynamics (Gini Coefficient)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Simulation Round", fontsize=12)
    axes[0].set_ylabel("Gini Coefficient (0 = Equal, 1 = Maximal Inequality)", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.0)

    # --- Plot 2: Cooperation Rate ---
    axes[1].plot(trends_a["round_id"], trends_a["cooperation_rate"], label="Condition A (Ablated LLM)", color="crimson", linewidth=2.5, linestyle="--")
    axes[1].plot(trends_b["round_id"], trends_b["cooperation_rate"], label="Condition B (Grounded ESS LLM)", color="royalblue", linewidth=2.5)
    
    axes[1].set_title("Social Cohesion (Cooperation Rate)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Simulation Round", fontsize=12)
    axes[1].set_ylabel("Fraction of 'Cooperate' Actions", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    
    # Save Output
    out_dir = Path("analysis/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "phase_c_macro_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\nSuccess! Plot saved to: {out_file}")

if __name__ == "__main__":
    main()