from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRIMARY_ORDER = [
    "auditable_random_ess_persona",
    "pure_llm_ess_persona",
    "grounded_llm_ess_persona",
]

PERSONA_SENSITIVITY_ORDER = [
    "pure_llm_synth_persona",
    "pure_llm_ess_persona",
    "grounded_llm_synth_persona",
    "grounded_llm_ess_persona",
]

DISPLAY_NAMES = {
    "auditable_random_ess_persona": "Auditable Random\n(ESS Persona)",
    "pure_llm_ess_persona": "Pure LLM\n(ESS Persona)",
    "grounded_llm_ess_persona": "LLM + ESS\n(ESS Persona)",
    "pure_llm_synth_persona": "Pure LLM\n(Synth Persona)",
    "grounded_llm_synth_persona": "LLM + ESS\n(Synth Persona)",
}

COLORS = {
    "auditable_random_ess_persona": "#4C78A8",
    "pure_llm_ess_persona": "#8F63D2",
    "grounded_llm_ess_persona": "#E45756",
    "pure_llm_synth_persona": "#72B7B2",
    "grounded_llm_synth_persona": "#F58518",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot grounding comparison results.")
    parser.add_argument("--registry", type=str, default="experiments/grounding_matrix_registry.json")
    parser.add_argument("--fig-dir", type=str, default="analysis/figures")
    parser.add_argument("--table-dir", type=str, default="analysis/tables")
    parser.add_argument("--report-dir", type=str, default="analysis/reports")
    return parser.parse_args()


def load_metrics(registry_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = []
    lorenz_records = []

    for entry in registry:
        run_dir = Path(entry["run_dir"])
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        wealth = summary["wealth"]
        stress = summary["stress"]
        behavior = summary["event_behavior"]

        rows.append(
            {
                "experiment_id": entry["experiment_id"],
                "condition_key": entry["condition_key"],
                "seed": entry["seed"],
                "is_primary": entry["is_primary"],
                "policy_family": entry["policy_family"],
                "persona_source": entry["persona_source"],
                "use_population_context": entry["use_population_context"],
                "wealth_mean": wealth["mean"],
                "wealth_gini": wealth["gini"],
                "stress_mean": stress["mean"],
                "work_rate": behavior.get("work_rate", 0.0),
                "save_rate": behavior.get("save_rate", 0.0),
                "cooperation_rate": behavior.get("cooperation_rate", 0.0),
                "rejected_action_rate": behavior.get("rejected_action_rate", 0.0),
                "run_dir": str(run_dir),
            }
        )

        lorenz_records.append(
            {
                "experiment_id": entry["experiment_id"],
                "condition_key": entry["condition_key"],
                "seed": entry["seed"],
                "population_share": wealth["lorenz_curve"]["population_share"],
                "value_share": wealth["lorenz_curve"]["value_share"],
                "gini": wealth["gini"],
            }
        )

    return pd.DataFrame(rows), lorenz_records


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["condition_key"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            wealth_mean=("wealth_mean", "mean"),
            wealth_mean_std=("wealth_mean", "std"),
            wealth_gini=("wealth_gini", "mean"),
            wealth_gini_std=("wealth_gini", "std"),
            stress_mean=("stress_mean", "mean"),
            stress_mean_std=("stress_mean", "std"),
            work_rate=("work_rate", "mean"),
            work_rate_std=("work_rate", "std"),
            save_rate=("save_rate", "mean"),
            save_rate_std=("save_rate", "std"),
            cooperation_rate=("cooperation_rate", "mean"),
            cooperation_rate_std=("cooperation_rate", "std"),
        )
    )
    return summary.fillna(0.0)


def _style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#1F2937",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def _condition_values(summary_df: pd.DataFrame, condition_order: list[str], metric: str, metric_std: str) -> tuple[list[float], list[float], list[str], list[str]]:
    means = []
    stds = []
    labels = []
    colors = []
    for key in condition_order:
        row = summary_df[summary_df["condition_key"] == key]
        if row.empty:
            continue
        means.append(float(row.iloc[0][metric]))
        stds.append(float(row.iloc[0][metric_std]))
        labels.append(DISPLAY_NAMES.get(key, key))
        colors.append(COLORS.get(key, "#6B7280"))
    return means, stds, labels, colors


def plot_primary_action_rates(summary_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics = [
        ("work_rate", "work_rate_std", "Work"),
        ("save_rate", "save_rate_std", "Save"),
        ("cooperation_rate", "cooperation_rate_std", "Cooperate"),
    ]
    condition_order = PRIMARY_ORDER

    x = range(len(condition_order))
    width = 0.22
    offsets = [-width, 0, width]

    for offset, (metric, metric_std, label) in zip(offsets, metrics):
        means, stds, _, colors = _condition_values(summary_df, condition_order, metric, metric_std)
        ax.bar(
            [i + offset for i in range(len(means))],
            means,
            width=width,
            yerr=stds,
            label=label,
            color=colors,
            alpha=0.9,
            capsize=4,
            edgecolor="white",
        )

    labels = [DISPLAY_NAMES[k] for k in condition_order if k in set(summary_df["condition_key"])]
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Primary Comparison: Action Mix")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_primary_key_metrics(summary_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        ("wealth_mean", "wealth_mean_std", "Mean Wealth"),
        ("wealth_gini", "wealth_gini_std", "Gini"),
        ("stress_mean", "stress_mean_std", "Mean Stress"),
    ]

    labels = [DISPLAY_NAMES[k] for k in PRIMARY_ORDER if k in set(summary_df["condition_key"])]
    colors = [COLORS[k] for k in PRIMARY_ORDER if k in set(summary_df["condition_key"])]

    for ax, (metric, metric_std, title) in zip(axes, panels):
        means, stds, _, _ = _condition_values(summary_df, PRIMARY_ORDER, metric, metric_std)
        ax.bar(labels, means, yerr=stds, color=colors, capsize=4, edgecolor="white")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Primary Comparison: Outcome Metrics", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_persona_sensitivity(summary_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("work_rate", "work_rate_std", "Work Rate"),
        ("cooperation_rate", "cooperation_rate_std", "Cooperation Rate"),
        ("wealth_gini", "wealth_gini_std", "Gini"),
    ]

    labels = [DISPLAY_NAMES[k] for k in PERSONA_SENSITIVITY_ORDER if k in set(summary_df["condition_key"])]
    colors = [COLORS[k] for k in PERSONA_SENSITIVITY_ORDER if k in set(summary_df["condition_key"])]

    for ax, (metric, metric_std, title) in zip(axes, panels):
        means, stds, _, _ = _condition_values(summary_df, PERSONA_SENSITIVITY_ORDER, metric, metric_std)
        ax.bar(labels, means, yerr=stds, color=colors, capsize=4, edgecolor="white")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("LLM Persona Sensitivity: Prompt Grounding vs Persona Source", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_lorenz_supplement(summary_df: pd.DataFrame, lorenz_records: list[dict], out_path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 8))

    for condition_key in PRIMARY_ORDER:
        rows = summary_df[summary_df["condition_key"] == condition_key]
        if rows.empty:
            continue
        mean_gini = float(rows.iloc[0]["wealth_gini"])
        candidates = [r for r in lorenz_records if r["condition_key"] == condition_key]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda r: abs(r["gini"] - mean_gini))
        ax.plot(
            chosen["population_share"],
            chosen["value_share"],
            label=f"{DISPLAY_NAMES.get(condition_key, condition_key)} (G={chosen['gini']:.3f})",
            color=COLORS.get(condition_key, "#6B7280"),
            linewidth=2.5,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.5, label="Perfect Equality")
    ax.set_title("Supplementary: Lorenz Curves")
    ax.set_xlabel("Cumulative Population Share")
    ax.set_ylabel("Cumulative Wealth Share")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    primary_df = summary_df[summary_df["condition_key"].isin(PRIMARY_ORDER)].copy()
    primary_df["label"] = primary_df["condition_key"].map(DISPLAY_NAMES)

    lines = [
        "# Grounding Comparison Report",
        "",
        "Main figures intentionally focus on `Auditable Random`, `Pure LLM`, and `LLM + ESS`.",
        "Template and rule-based baselines are omitted from the main figures because prior comparisons showed weak separation and limited explanatory value.",
        "",
        "## Primary Summary",
        "",
        "| Condition | Seeds | Wealth Mean | Gini | Stress Mean | Work Rate | Save Rate | Cooperation Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in primary_df.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['seeds'])} | "
            f"{row['wealth_mean']:.3f} | {row['wealth_gini']:.3f} | {row['stress_mean']:.3f} | "
            f"{row['work_rate']:.3f} | {row['save_rate']:.3f} | {row['cooperation_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Pure LLM` here means no population grounding, no social context, no memory context, and no balancing hint.",
            "- `LLM + ESS` includes ESS-derived population grounding plus social context and memory.",
            "- `Auditable Random` is a weighted stochastic baseline with deterministic seeds and per-step audit logs.",
            "- Lorenz curves are generated as supplementary material only.",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    table_dir = Path(args.table_dir)
    report_dir = Path(args.report_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df, lorenz_records = load_metrics(Path(args.registry))
    if df.empty:
        raise RuntimeError("No experiment summaries found in registry.")

    summary_df = summarize(df)
    df.to_csv(table_dir / "grounding_comparison_seed_metrics.csv", index=False)
    summary_df.to_csv(table_dir / "grounding_comparison_condition_summary.csv", index=False)

    plot_primary_action_rates(summary_df, fig_dir / "grounding_primary_action_rates.png")
    plot_primary_key_metrics(summary_df, fig_dir / "grounding_primary_key_metrics.png")
    plot_persona_sensitivity(summary_df, fig_dir / "grounding_persona_sensitivity.png")
    plot_lorenz_supplement(summary_df, lorenz_records, fig_dir / "grounding_lorenz_supplement.png")
    write_markdown_report(summary_df, report_dir / "grounding_comparison_report.md")

    print("Saved:")
    print(fig_dir / "grounding_primary_action_rates.png")
    print(fig_dir / "grounding_primary_key_metrics.png")
    print(fig_dir / "grounding_persona_sensitivity.png")
    print(fig_dir / "grounding_lorenz_supplement.png")
    print(table_dir / "grounding_comparison_seed_metrics.csv")
    print(table_dir / "grounding_comparison_condition_summary.csv")
    print(report_dir / "grounding_comparison_report.md")


if __name__ == "__main__":
    main()
