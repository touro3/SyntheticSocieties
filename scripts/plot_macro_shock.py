import argparse
import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, repo_root)

import matplotlib.pyplot as plt
import seaborn as sns
from metrics.macro_metrics import SocietyMacroMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="experiments/macro_shock")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trends_a = SocietyMacroMetrics.analyze_trajectory(str(data_dir / "condition_a_shock.parquet")).to_pandas()
    trends_b = SocietyMacroMetrics.analyze_trajectory(str(data_dir / "condition_b_shock.parquet")).to_pandas()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Plot Wealth Shock
    axes[0].plot(trends_a["round_id"], trends_a["total_wealth"], label="Cond A (Ablated)", color="crimson", linewidth=2.5, linestyle="--")
    axes[0].plot(trends_b["round_id"], trends_b["total_wealth"], label="Cond B (Grounded)", color="royalblue", linewidth=2.5)
    axes[0].axvline(x=15, color='black', linestyle=':', linewidth=2, label="Market Crash")
    axes[0].set_title("Macroeconomic Wealth Trajectory", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("Total Societal Wealth", fontsize=12)
    axes[0].legend()

    # Plot Cooperation Panic
    axes[1].plot(trends_a["round_id"], trends_a["cooperation_rate"], label="Cond A (Ablated)", color="crimson", linewidth=2.5, linestyle="--")
    axes[1].plot(trends_b["round_id"], trends_b["cooperation_rate"], label="Cond B (Grounded)", color="royalblue", linewidth=2.5)
    axes[1].axvline(x=15, color='black', linestyle=':', linewidth=2, label="Market Crash")
    axes[1].set_title("Crisis Response (Cooperation Rate)", fontsize=15, fontweight="bold")
    axes[1].set_ylabel("Fraction of 'Cooperate' Actions", fontsize=12)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    plt.tight_layout()
    out_file = Path("analysis/figures/macro_shock_resilience.png")
    plt.savefig(out_file, dpi=400, bbox_inches="tight")
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    main()
