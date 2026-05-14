"""Fit trust-band-specific cooperation models within the Austrian ESS sample.

Partially mitigates Limitation #13 (Austrian-only baseline) by producing
per-trust-band logistic regressions so the persona-fidelity metric can use a
baseline calibrated to the *trust profile* of each cluster, even though the
underlying respondents remain Austrian.

The Austrian sample is partitioned by `trust_people` into the same six
trust bands used in cross-cultural validation (eastern, southern, western,
anglo, northern, nordic — see `data/cross_cultural_benchmarks_expanded.json`).
A logistic regression is fitted on each band, plus a pooled-AT control.

Output: `data/cooperation_model_per_band.json`

This does NOT remove the Austrian-attitude confound but is the strongest
within-data robustness check available without additional ESS country files.
The full multi-country refit remains pending against the multi-country MD release.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
FEATURES = [
    "trust_people",
    "trust_fairness",
    "trust_helpfulness",
    "risk_taking",
    "social_meeting_freq",
    "social_activity",
    "reduce_inequality",
]
TARGET = "volunteered"
BENCH_PATH = REPO / "data" / "cross_cultural_benchmarks_expanded.json"
OUT_PATH = REPO / "data" / "cooperation_model_per_band.json"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["coop"] = (df[TARGET] == 1).astype(int)
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df.dropna(subset=["coop"])


def fit_band(df_band: pd.DataFrame, C: float = 1.0, n_folds: int = 5) -> dict:
    if len(df_band) < 40 or df_band["coop"].sum() < 5:
        return {
            "n_samples": int(len(df_band)),
            "n_positive": int(df_band["coop"].sum()),
            "auc": None,
            "skipped_reason": "insufficient sample for stable fit",
        }
    X = df_band[FEATURES].values.astype(float)
    y = df_band["coop"].values.astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=1000, random_state=0))])
    n_folds_eff = min(n_folds, int(y.sum()), int((1 - y).sum()))
    auc = None
    if n_folds_eff >= 3:
        cv = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, y_prob))
        brier = float(brier_score_loss(y, y_prob))
    else:
        brier = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]
    return {
        "n_samples": int(len(df_band)),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()),
        "auc_cv": auc,
        "brier_cv": brier,
        "cv_folds": n_folds_eff,
        "feature_means": scaler.mean_.tolist(),
        "feature_stds": scaler.scale_.tolist(),
        "coef_standardized": lr.coef_[0].tolist(),
        "intercept_standardized": float(lr.intercept_[0]),
        "coef_original": (lr.coef_[0] / scaler.scale_).tolist(),
        "intercept_original": float(lr.intercept_[0] - np.dot(lr.coef_[0], scaler.mean_ / scaler.scale_)),
    }


def main() -> None:
    bench = json.loads(BENCH_PATH.read_text())["clusters"]
    df = prepare(pd.read_parquet(REPO / "data" / "ess_clean.parquet"))
    print(f"Loaded {len(df)} AT respondents (positive rate {df['coop'].mean():.1%}).")

    results: dict[str, dict] = {}
    results["pooled_at"] = fit_band(df)
    print(
        f"\n[pooled_at]            n={results['pooled_at']['n_samples']:>4} "
        f"AUC={results['pooled_at']['auc_cv'] or float('nan'):.3f} "
        f"base={results['pooled_at']['base_rate']:.1%}"
    )

    print()
    print(f"{'cluster':<12} {'trust_lo':>8} {'trust_hi':>8} {'n':>5} {'pos':>5} {'AUC':>6} {'base':>6}")
    print("-" * 55)
    for name, meta in bench.items():
        lo, hi = float(meta["trust_lo"]), float(meta["trust_hi"])
        sub = df[(df["trust_people"] >= lo) & (df["trust_people"] < hi)]
        fit = fit_band(sub)
        fit["trust_lo"] = lo
        fit["trust_hi"] = hi
        fit["cluster_countries"] = meta["countries"]
        fit["cluster_ess_mean_trust"] = meta["ess_mean_trust_people"]
        results[name] = fit
        auc_str = f"{fit['auc_cv']:.3f}" if fit.get("auc_cv") else "  —  "
        base_str = f"{fit.get('base_rate', 0):.1%}" if "base_rate" in fit else "  —  "
        print(
            f"{name:<12} {lo:>8.2f} {hi:>8.2f} {fit['n_samples']:>5} "
            f"{fit.get('n_positive', 0):>5} {auc_str:>6} {base_str:>6}"
        )

    output = {
        "meta": {
            "description": (
                "Per-trust-band cooperation models fitted on Austrian ESS R11. "
                "Partial mitigation of the Austrian-only baseline confound: each "
                "cross-cultural cluster is now scored against a baseline calibrated "
                "to AT respondents at the matching trust level, rather than the "
                "global AT mean. Multi-country MD release is still the proper fix."
            ),
            "ess_source": "data/ess_clean.parquet",
            "ess_country": "AT (partitioned by trust_people)",
            "ess_round": 11,
            "features": FEATURES,
            "target_variable": TARGET,
            "bands_source": "data/cross_cultural_benchmarks_expanded.json",
            "known_limitations": [
                "All respondents Austrian; only the trust *profile* varies across bands",
                "Some bands have small n (Nordic-equivalent AT band typically <50)",
                "True cross-country fit requires multi-country ESS R11 MD release",
            ],
            "audit_row": "limitation_13_partial_resolution",
        },
        "bands": results,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Per-band models → {OUT_PATH}")


if __name__ == "__main__":
    main()
