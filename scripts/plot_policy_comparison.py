import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
    table_path = Path("analysis/tables/policy_comparison.parquet")
    output_path = Path("analysis/figures/policy_wealth_mean.png")

    if not table_path.exists():
        print("Comparison table does not exist. Run export_policy_comparison.py first.")
        return

    df = pl.read_parquet(table_path).sort("policy_type")
    pdf = df.to_pandas()

    plt.figure(figsize=(8, 5))
    plt.bar(
        pdf["policy_type"],
        pdf["wealth_mean_avg"],
        yerr=pdf["wealth_mean_std"].fillna(0.0),
        capsize=5,
    )
    plt.title("Average Wealth Mean by Policy")
    plt.xlabel("Policy Type")
    plt.ylabel("Average Wealth Mean")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
