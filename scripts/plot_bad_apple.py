import argparse
import sys
from pathlib import Path
    
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import matplotlib.pyplot as plt
import seaborn as sns
from metrics.macro_metrics import SocietyMacroMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="experiments/bad_apple")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    file_a = data_dir / "condition_a_adversarial.parquet"
    file_b = data_dir / "condition_b_adversarial.parquet"

    print("Extracting macro trajectories and calculating adversarial impact...")
    trends_a = SocietyMacroMetrics.analyze_trajectory(str(file_a)).to_pandas()
    trends_b = SocietyMacroMetrics.analyze_trajectory(str(file_b)).to_pandas()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].plot(trends_a["round_id"], trends_a["gini_coefficient"], label="Condition A (Ablated LLM + 5% Adversaries)", color="crimson", linewidth=2.5, linestyle="--")
    axes[0].plot(trends_b["round_id"], trends_b["gini_coefficient"], label="Condition B (Grounded BGF + 5% Adversaries)", color="royalblue", linewidth=2.5)
    
    axes[0].set_title("Economic Vulnerability (Gini Coefficient)", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Simulation Round", fontsize=12)
    axes[0].set_ylabel("Inequality (0 = Perfect Equality, 1 = Max Inequality)", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.0)

    axes[1].plot(trends_a["round_id"], trends_a["cooperation_rate"], label="Condition A (Ablated LLM + 5% Adversaries)", color="crimson", linewidth=2.5, linestyle="--")
    axes[1].plot(trends_b["round_id"], trends_b["cooperation_rate"], label="Condition B (Grounded BGF + 5% Adversaries)", color="royalblue", linewidth=2.5)
    
    axes[1].set_title("Social Immunity (Global Cooperation Rate)", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Simulation Round", fontsize=12)
    axes[1].set_ylabel("Fraction of 'Cooperate' Actions", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    
    out_dir = Path("analysis/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bad_apple_resilience.png"
    plt.savefig(out_file, dpi=400, bbox_inches="tight")
    print(f"\nSuccess! High-resolution plot saved to: {out_file}")

if __name__ == "__main__":
    main()
