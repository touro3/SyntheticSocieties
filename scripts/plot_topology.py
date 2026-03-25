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
    parser.add_argument("--data-dir", type=str, default="experiments/topology_test")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trends_a = SocietyMacroMetrics.analyze_trajectory(str(data_dir / "condition_a_dense.parquet")).to_pandas()
    trends_b = SocietyMacroMetrics.analyze_trajectory(str(data_dir / "condition_b_dense.parquet")).to_pandas()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].plot(trends_a["round_id"], trends_a["gini_coefficient"], label="Cond A (Ablated - Dense)", color="crimson", linewidth=2.5, linestyle="--")
    axes[0].plot(trends_b["round_id"], trends_b["gini_coefficient"], label="Cond B (Grounded - Dense)", color="royalblue", linewidth=2.5)
    axes[0].set_title("Gini under Topological Dictatorship", fontsize=15, fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend()

    axes[1].plot(trends_a["round_id"], trends_a["cooperation_rate"], label="Cond A (Ablated - Dense)", color="crimson", linewidth=2.5, linestyle="--")
    axes[1].plot(trends_b["round_id"], trends_b["cooperation_rate"], label="Cond B (Grounded - Dense)", color="royalblue", linewidth=2.5)
    axes[1].set_title("Trust Collapse in Fully Connected Network", fontsize=15, fontweight="bold")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    plt.tight_layout()
    out_file = Path("analysis/figures/topology_dictatorship.png")
    plt.savefig(out_file, dpi=400, bbox_inches="tight")
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    main()
