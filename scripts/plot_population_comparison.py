from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def main() -> None:
    table_path = Path("analysis/tables/population_comparison.parquet")
    output_path = Path("analysis/figures/population_wealth_mean.png")

    if not table_path.exists():
        print("Comparison table does not exist. Run export_population_comparison.py first.")
        return

    df = pl.read_parquet(table_path).sort(["policy_type", "population_size"])
    pdf = df.to_pandas()

    policies = sorted(pdf["policy_type"].unique())
    populations = sorted(pdf["population_size"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.25
    x = range(len(policies))

    for i, pop_size in enumerate(populations):
        subset = pdf[pdf["population_size"] == pop_size].set_index("policy_type").reindex(policies)

        positions = [p + (i - (len(populations) - 1) / 2) * width for p in x]

        ax.bar(
            positions,
            subset["wealth_mean_avg"],
            width=width,
            yerr=subset["wealth_mean_std"].fillna(0.0),
            capsize=4,
            label=f"n={pop_size}",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(policies)
    ax.set_title("Average Wealth Mean by Policy and Population Size")
    ax.set_xlabel("Policy Type")
    ax.set_ylabel("Average Wealth Mean")
    ax.legend(title="Population Size")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
