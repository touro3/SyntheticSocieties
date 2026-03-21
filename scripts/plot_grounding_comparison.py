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
    "grounded_llm_ess_persona": "Grounded LLM\n(ESS Persona)",
    "pure_llm_synth_persona": "Pure LLM\n(Synth Persona)",
    "grounded_llm_synth_persona": "Grounded LLM\n(Synth Persona)",
}

CONDITION_COLORS = {
    "auditable_random_ess_persona": "#4C78A8",
    "pure_llm_ess_persona": "#8F63D2",
    "grounded_llm_ess_persona": "#E45756",
    "pure_llm_synth_persona": "#72B7B2",
    "grounded_llm_synth_persona": "#F58518",
}

ACTION_COLORS = {
    "work_rate": "#4C78A8",
    "save_rate": "#F2A541",
    "cooperation_rate": "#2CB1A1",
}

ACTION_LABELS = {
    "work_rate": "Work",
    "save_rate": "Save",
    "cooperation_rate": "Cooperate",
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
                "use_social_context": entry["use_social_context"],
                "use_memory_context": entry["use_memory_context"],
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
        colors.append(CONDITION_COLORS.get(key, "#6B7280"))
    return means, stds, labels, colors


def plot_primary_action_rates(summary_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11, 6))

    available_conditions = [k for k in PRIMARY_ORDER if k in set(summary_df["condition_key"])]
    labels = [DISPLAY_NAMES[k] for k in available_conditions]

    x = list(range(len(available_conditions)))
    width = 0.22
    metrics = [
        ("work_rate", "work_rate_std"),
        ("save_rate", "save_rate_std"),
        ("cooperation_rate", "cooperation_rate_std"),
    ]
    offsets = [-width, 0.0, width]

    for offset, (metric, metric_std) in zip(offsets, metrics):
        means, stds, _, _ = _condition_values(summary_df, available_conditions, metric, metric_std)
        ax.bar(
            [i + offset for i in x],
            means,
            width=width,
            yerr=stds,
            color=ACTION_COLORS[metric],
            label=ACTION_LABELS[metric],
            alpha=0.92,
            capsize=4,
            edgecolor="white",
            linewidth=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Primary Comparison: Action Mix")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="Action")
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
    colors = [CONDITION_COLORS[k] for k in PRIMARY_ORDER if k in set(summary_df["condition_key"])]

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("work_rate", "work_rate_std", "Work Rate"),
        ("cooperation_rate", "cooperation_rate_std", "Cooperation Rate"),
        ("wealth_gini", "wealth_gini_std", "Gini"),
    ]

    labels = [DISPLAY_NAMES[k] for k in PERSONA_SENSITIVITY_ORDER if k in set(summary_df["condition_key"])]
    colors = [CONDITION_COLORS[k] for k in PERSONA_SENSITIVITY_ORDER if k in set(summary_df["condition_key"])]

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
            color=CONDITION_COLORS.get(condition_key, "#6B7280"),
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


def _metric(summary_df: pd.DataFrame, key: str, metric: str) -> float:
    row = summary_df[summary_df["condition_key"] == key]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])


def write_markdown_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    primary_df = summary_df[summary_df["condition_key"].isin(PRIMARY_ORDER)].copy()
    primary_df["label"] = primary_df["condition_key"].map(DISPLAY_NAMES)

    random_work = _metric(summary_df, "auditable_random_ess_persona", "work_rate")
    pure_work = _metric(summary_df, "pure_llm_ess_persona", "work_rate")
    grounded_work = _metric(summary_df, "grounded_llm_ess_persona", "work_rate")

    random_save = _metric(summary_df, "auditable_random_ess_persona", "save_rate")
    pure_save = _metric(summary_df, "pure_llm_ess_persona", "save_rate")
    grounded_save = _metric(summary_df, "grounded_llm_ess_persona", "save_rate")

    random_coop = _metric(summary_df, "auditable_random_ess_persona", "cooperation_rate")
    pure_coop = _metric(summary_df, "pure_llm_ess_persona", "cooperation_rate")
    grounded_coop = _metric(summary_df, "grounded_llm_ess_persona", "cooperation_rate")

    pure_wealth = _metric(summary_df, "pure_llm_ess_persona", "wealth_mean")
    grounded_wealth = _metric(summary_df, "grounded_llm_ess_persona", "wealth_mean")
    random_wealth = _metric(summary_df, "auditable_random_ess_persona", "wealth_mean")

    pure_gini = _metric(summary_df, "pure_llm_ess_persona", "wealth_gini")
    grounded_gini = _metric(summary_df, "grounded_llm_ess_persona", "wealth_gini")
    random_gini = _metric(summary_df, "auditable_random_ess_persona", "wealth_gini")

    pure_stress = _metric(summary_df, "pure_llm_ess_persona", "stress_mean")
    grounded_stress = _metric(summary_df, "grounded_llm_ess_persona", "stress_mean")
    random_stress = _metric(summary_df, "auditable_random_ess_persona", "stress_mean")

    lines = [
        "# Grounding Comparison Report",
        "",
        "Main figures focus on `Auditable Random`, `Pure LLM`, and `Grounded LLM`.",
        "Template and rule-based baselines are omitted from the main figures because earlier comparisons showed weak separation and limited explanatory value.",
        "",
        "## Primary Summary",
        "",
        "| Condition | Seeds | Wealth Mean | Gini | Stress Mean | Work Rate | Save Rate | Cooperation Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    ordered_rows = [
        "auditable_random_ess_persona",
        "pure_llm_ess_persona",
        "grounded_llm_ess_persona",
    ]

    for key in ordered_rows:
        row = summary_df[summary_df["condition_key"] == key]
        if row.empty:
            continue
        row = row.iloc[0]
        label = DISPLAY_NAMES.get(key, key).replace("\n", " ")
        lines.append(
            f"| {label} | {int(row['seeds'])} | "
            f"{row['wealth_mean']:.3f} | {row['wealth_gini']:.3f} | {row['stress_mean']:.3f} | "
            f"{row['work_rate']:.3f} | {row['save_rate']:.3f} | {row['cooperation_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The main behavioral shift is not in work propensity. Work remains in a relatively narrow band across the primary conditions: `Auditable Random={random_work:.3f}`, `Pure LLM={pure_work:.3f}`, and `Grounded LLM={grounded_work:.3f}`.",
            "",
            f"The strongest difference appears in how non-work behavior is allocated between saving and cooperation. `Pure LLM` strongly favors saving (`save_rate={pure_save:.3f}`) and nearly eliminates cooperation (`cooperation_rate={pure_coop:.3f}`). By contrast, `Grounded LLM` sharply reduces saving (`save_rate={grounded_save:.3f}`) and increases cooperation (`cooperation_rate={grounded_coop:.3f}`). `Auditable Random` remains behaviorally mixed, with `save_rate={random_save:.3f}` and `cooperation_rate={random_coop:.3f}`.",
            "",
            f"In short-horizon economic terms, `Pure LLM` currently performs best on mean wealth (`{pure_wealth:.3f}`) and lowest inequality (`Gini={pure_gini:.3f}`). `Grounded LLM` is more cooperative, but that increase in cooperation does not yet convert into lower inequality or lower stress over 5 rounds (`wealth_mean={grounded_wealth:.3f}`, `Gini={grounded_gini:.3f}`, `stress_mean={grounded_stress:.3f}`). `Auditable Random` remains the lowest-stress baseline (`stress_mean={random_stress:.3f}`) while preserving a balanced action mix.",
            "",
            "These results suggest that, in the current environment, cooperation is behaviorally meaningful but not yet rewarded quickly enough to outperform self-preserving strategies on short-horizon wealth and stress metrics.",
            "",
            "## Caveats",
            "",
            "- These comparisons are based on 3 seeds, so uncertainty estimates are still coarse.",
            "- `Pure LLM` here means no population grounding, no social context, no memory context, and no balancing hint.",
            "- `Grounded LLM` includes ESS-derived population grounding plus social context and memory, so it should not be interpreted as an ESS-only effect.",
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
