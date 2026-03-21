from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics.recalibration import (
    apply_affine,
    apply_variance_scaling,
    fit_affine,
    fit_variance_scaling,
)


def composite_score(frame: pd.DataFrame, target_items: list[dict[str, Any]], prefix: str = "") -> pd.Series:
    cols = []
    for item in target_items:
        name = f"{prefix}{item['name']}"
        series = frame[name].astype(float).clip(0.0, 1.0)
        if item.get("inverse", False):
            series = 1.0 - series
        cols.append(series)
    return pd.concat(cols, axis=1).mean(axis=1) * 100.0


def summarize_synthetic_runs(
    synthetic_runs_df: pd.DataFrame,
    target_items: list[dict[str, Any]],
) -> pd.DataFrame:
    agg = {"replication_seed": "count"}
    for item in target_items:
        agg[item["name"]] = "mean"

    grouped = synthetic_runs_df.groupby("profile_id", as_index=False).agg(agg).rename(columns={"replication_seed": "n_synthetic"})
    grouped["synthetic_score_0_100"] = composite_score(grouped, target_items)
    return grouped


def compute_pca_projection(real_profiles: pd.DataFrame, synthetic_profiles: pd.DataFrame, target_items: list[dict[str, Any]]) -> dict:
    target_names = [f"real_{item['name']}" for item in target_items]
    synth_names = [item["name"] for item in target_items]

    X_real = real_profiles[target_names].to_numpy(dtype=float)
    X_synth = synthetic_profiles[synth_names].to_numpy(dtype=float)

    mu = X_real.mean(axis=0)
    sigma = X_real.std(axis=0, ddof=1)
    sigma = np.where(sigma == 0, 1.0, sigma)

    X_real_std = (X_real - mu) / sigma
    X_synth_std = (X_synth - mu) / sigma

    _, _, vt = np.linalg.svd(X_real_std, full_matrices=False)
    pc1 = vt[0]

    real_pc1 = X_real_std @ pc1
    synth_pc1 = X_synth_std @ pc1

    return {
        "explained_variance_pc1_proxy": float(np.var(real_pc1) / max(np.var(X_real_std, axis=0).sum(), 1e-9)),
        "real_pc1": real_pc1.tolist(),
        "synthetic_pc1": synth_pc1.tolist(),
        "pc1_pearson": float(pd.Series(real_pc1).corr(pd.Series(synth_pc1), method="pearson")),
        "pc1_spearman": float(pd.Series(real_pc1).corr(pd.Series(synth_pc1), method="spearman")),
        "pc1_bias": float(np.mean(synth_pc1 - real_pc1)),
    }


def compute_fidelity_report(
    real_profiles_df: pd.DataFrame,
    synthetic_profile_df: pd.DataFrame,
    target_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = real_profiles_df.merge(
        synthetic_profile_df,
        on="profile_id",
        how="inner",
        validate="one_to_one",
    ).copy()

    merged["real_score_0_100"] = composite_score(merged, target_items, prefix="real_")
    merged["synthetic_score_0_100"] = composite_score(merged, target_items, prefix="")
    merged["score_diff"] = merged["synthetic_score_0_100"] - merged["real_score_0_100"]
    merged["abs_score_diff"] = merged["score_diff"].abs()

    real_scores = merged["real_score_0_100"].to_numpy(dtype=float)
    synth_scores = merged["synthetic_score_0_100"].to_numpy(dtype=float)

    affine = fit_affine(real_scores, synth_scores)
    scaled = fit_variance_scaling(real_scores, synth_scores)

    calibrated_affine = apply_affine(synth_scores, affine["slope"], affine["intercept"])
    calibrated_scaled = apply_variance_scaling(
        synth_scores,
        scale=scaled["scale"],
        synthetic_mean=scaled["synthetic_mean"],
        real_mean=scaled["real_mean"],
    )

    item_metrics = {}
    for item in target_items:
        real_col = f"real_{item['name']}"
        synth_col = item["name"]
        diff = merged[synth_col] - merged[real_col]
        item_metrics[item["name"]] = {
            "mae": float(diff.abs().mean()),
            "bias": float(diff.mean()),
            "pearson": float(merged[real_col].corr(merged[synth_col], method="pearson")),
            "spearman": float(merged[real_col].corr(merged[synth_col], method="spearman")),
        }
        merged[f"diff_{item['name']}"] = diff

    pca = compute_pca_projection(real_profiles_df, synthetic_profile_df, target_items)

    report = {
        "n_profiles": int(len(merged)),
        "score_metrics": {
            "mean_real_score": float(real_scores.mean()),
            "mean_synthetic_score": float(synth_scores.mean()),
            "score_bias": float((synth_scores - real_scores).mean()),
            "score_mae": float(np.abs(synth_scores - real_scores).mean()),
            "score_rmse": float(np.sqrt(np.mean((synth_scores - real_scores) ** 2))),
            "dispersion_ratio": float(np.std(synth_scores, ddof=1) / max(np.std(real_scores, ddof=1), 1e-9)),
            "pearson": float(pd.Series(real_scores).corr(pd.Series(synth_scores), method="pearson")),
            "spearman": float(pd.Series(real_scores).corr(pd.Series(synth_scores), method="spearman")),
        },
        "item_metrics": item_metrics,
        "pca_metrics": pca,
        "affine_recalibration": {
            **affine,
            "mae_after": float(np.mean(np.abs(calibrated_affine - real_scores))),
            "bias_after": float(np.mean(calibrated_affine - real_scores)),
        },
        "variance_scaling": {
            **scaled,
            "mae_after": float(np.mean(np.abs(calibrated_scaled - real_scores))),
            "bias_after": float(np.mean(calibrated_scaled - real_scores)),
        },
    }

    return report, merged.sort_values("profile_id").reset_index(drop=True)


def write_report_files(
    run_dir: str | Path,
    report: dict[str, Any],
    per_profile_df: pd.DataFrame,
    synthetic_profile_df: pd.DataFrame,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "fidelity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    per_profile_df.to_csv(run_dir / "per_profile_comparison.csv", index=False)
    synthetic_profile_df.to_csv(run_dir / "synthetic_profile_summary.csv", index=False)
