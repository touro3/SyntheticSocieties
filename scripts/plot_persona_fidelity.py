from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot persona fidelity benchmark outputs.")
    parser.add_argument("--run-dir", required=True, type=str)
    return parser.parse_args()


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
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def plot_score_scatter(per_profile_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        per_profile_df["real_score_0_100"],
        per_profile_df["synthetic_score_0_100"],
        color="#4C78A8",
        s=60,
        alpha=0.9,
    )
    mn = min(per_profile_df["real_score_0_100"].min(), per_profile_df["synthetic_score_0_100"].min())
    mx = max(per_profile_df["real_score_0_100"].max(), per_profile_df["synthetic_score_0_100"].max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="#9CA3AF", label="Perfect alignment")
    for _, row in per_profile_df.iterrows():
        ax.text(row["real_score_0_100"] + 0.3, row["synthetic_score_0_100"] + 0.3, row["profile_id"], fontsize=8)
    ax.set_xlabel("Real profile score (0-100)")
    ax.set_ylabel("Synthetic profile score (0-100)")
    ax.set_title("Persona Fidelity: Real vs Synthetic Profile Scores")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_profile_order(per_profile_df: pd.DataFrame, out_path: Path) -> None:
    _style()
    ordered = per_profile_df.sort_values("real_score_0_100").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(ordered))
    ax.plot(x, ordered["real_score_0_100"], marker="o", linewidth=2, label="Real", color="#2CB1A1")
    ax.plot(x, ordered["synthetic_score_0_100"], marker="o", linewidth=2, label="Synthetic", color="#E45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["profile_id"], rotation=45)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("Profile Ordering: Real vs Synthetic")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_item_mae(report: dict, out_path: Path) -> None:
    _style()
    item_metrics = report["item_metrics"]
    items = list(item_metrics.keys())
    maes = [item_metrics[item]["mae"] for item in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(items, maes, color="#8F63D2", edgecolor="white")
    ax.set_ylabel("MAE")
    ax.set_title("Item-Level Mean Absolute Error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown(run_dir: Path, report: dict, out_path: Path) -> None:
    score = report["score_metrics"]
    affine = report["affine_recalibration"]
    pca = report["pca_metrics"]

    lines = [
        "# Persona Fidelity Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Profiles: `{report['n_profiles']}`",
        "",
        "## Core Metrics",
        "",
        f"- Mean real score: `{score['mean_real_score']:.3f}`",
        f"- Mean synthetic score: `{score['mean_synthetic_score']:.3f}`",
        f"- Score bias: `{score['score_bias']:.3f}`",
        f"- Score MAE: `{score['score_mae']:.3f}`",
        f"- Dispersion ratio (synthetic / real): `{score['dispersion_ratio']:.3f}`",
        f"- Pearson correlation: `{score['pearson']:.3f}`",
        f"- Spearman correlation: `{score['spearman']:.3f}`",
        "",
        "## Latent-Space Check",
        "",
        f"- PC1 Pearson correlation: `{pca['pc1_pearson']:.3f}`",
        f"- PC1 Spearman correlation: `{pca['pc1_spearman']:.3f}`",
        f"- PC1 bias: `{pca['pc1_bias']:.3f}`",
        "",
        "## Recalibration",
        "",
        f"- Affine slope: `{affine['slope']:.3f}`",
        f"- Affine intercept: `{affine['intercept']:.3f}`",
        f"- MAE after affine recalibration: `{affine['mae_after']:.3f}`",
        f"- Bias after affine recalibration: `{affine['bias_after']:.3f}`",
        "",
        "## Interpretation",
        "",
        "This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.",
        "High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.",
        "Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    fig_dir = Path("analysis/figures")
    report_dir = Path("analysis/reports")
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((run_dir / "fidelity_report.json").read_text(encoding="utf-8"))
    per_profile_df = pd.read_csv(run_dir / "per_profile_comparison.csv")

    plot_score_scatter(per_profile_df, fig_dir / f"{run_dir.name}_score_scatter.png")
    plot_profile_order(per_profile_df, fig_dir / f"{run_dir.name}_profile_order.png")
    plot_item_mae(report, fig_dir / f"{run_dir.name}_item_mae.png")
    write_markdown(run_dir, report, report_dir / f"{run_dir.name}_persona_fidelity_report.md")

    print(fig_dir / f"{run_dir.name}_score_scatter.png")
    print(fig_dir / f"{run_dir.name}_profile_order.png")
    print(fig_dir / f"{run_dir.name}_item_mae.png")
    print(report_dir / f"{run_dir.name}_persona_fidelity_report.md")


if __name__ == "__main__":
    main()
