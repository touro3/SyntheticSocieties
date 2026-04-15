"""Fit an empirical cooperation rate model from ESS behavioral data.

This script replaces the heuristic formula
    expected_coop = 0.2 + 0.6 * trust_people * (1 - risk_tolerance)
with a logistic regression model fitted against real observed behavior from
the European Social Survey (ESS Round 11, Austria, N=866).

Target variable
---------------
``volunteered``: whether the respondent volunteered in the past year
(ESS variable volunap: 1=yes, 2=no). Volunteering is the best available
behavioral proxy for cooperation in the ESS — it is an observed action,
not an attitude, and maps directly to the altruistic wealth-transfer
mechanism in the BGF economy (cooperate: spend wealth to benefit others).

Limitation: only Austria is available in the cleaned ESS parquet.
Generalizability to other ESS countries is unknown. All model cards and
downstream citations should note this constraint.

Model
-----
Logistic regression with L2 regularization (sklearn), standardized inputs.
Features: trust_people, trust_fairness, trust_helpfulness, risk_taking,
          social_meeting_freq, social_activity, reduce_inequality
Validation: 10-fold stratified cross-validation (AUC, Brier score)
Uncertainty: 1000 bootstrap resamples for 95% CIs per coefficient
Calibration: reliability diagram bins

Output
------
data/cooperation_model.json — coefficients, intercept, feature means/stds,
    CV metrics, bootstrap CIs, calibration data.

Usage
-----
    python scripts/fit_cooperation_model.py
    python scripts/fit_cooperation_model.py --ess-path data/ess_clean.parquet
    python scripts/fit_cooperation_model.py --n-bootstrap 500 --cv-folds 5
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Configuration ────────────────────────────────────────────────────────────

FEATURES = [
    "trust_people",  # ESS: interpersonal trust (0-1)
    "trust_fairness",  # ESS: belief others play fair (0-1)
    "trust_helpfulness",  # ESS: belief others are helpful (0-1)
    "risk_taking",  # ESS: self-reported risk tolerance (0-1, reversed in profiles)
    "social_meeting_freq",  # ESS: frequency of social meetings (0-1)
    "social_activity",  # ESS: social engagement index (0-1)
    "reduce_inequality",  # ESS: support for redistribution (pro-social value)
]

TARGET = "volunteered"  # ESS: 1=volunteered, 2=did not volunteer
OUTPUT_PATH = Path("data/cooperation_model.json")


def load_and_prepare(ess_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load ESS parquet, recode target, impute features, return X, y, full df."""
    df = pd.read_parquet(ess_path)

    # Recode target: ESS codes 1=yes/2=no → binary 1/0
    if df[TARGET].isin([1, 2]).all():
        df["coop"] = (df[TARGET] == 1).astype(int)
    else:
        # Already 0/1 encoded
        df["coop"] = df[TARGET].astype(int)

    # Impute missing feature values with column median (conservative)
    # Only a handful of rows are affected (trust_fairness: 3, others: ≤1)
    for col in FEATURES:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"  Imputed {n_missing} missing values in '{col}' with median={median:.3f}")

    # Drop rows where target itself is missing
    df = df.dropna(subset=["coop"])

    X = df[FEATURES].values.astype(float)
    y = df["coop"].values.astype(int)

    return X, y, df


def cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, n_folds: int) -> dict:
    """10-fold stratified CV → AUC, Brier score, per-fold stats."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Collect predicted probabilities across folds
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)

    # Per-fold AUC for variance estimate
    fold_aucs = []
    for train_idx, val_idx in cv.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        p = pipeline.predict_proba(X[val_idx])[:, 1]
        if len(np.unique(y[val_idx])) > 1:
            fold_aucs.append(roc_auc_score(y[val_idx], p))

    return {
        "auc_mean": float(auc),
        "auc_per_fold": [float(a) for a in fold_aucs],
        "auc_std": float(np.std(fold_aucs)),
        "brier_score": float(brier),
        "n_folds": n_folds,
        "n_samples": int(len(y)),
        "base_rate": float(y.mean()),  # fraction who volunteered
    }


def bootstrap_coefficients(X: np.ndarray, y: np.ndarray, n_resamples: int, C: float) -> dict:
    """Bootstrap 95% CIs for logistic regression coefficients.

    Each resample: stratified sample with replacement, fit logistic regression,
    record standardized coefficients. Returns per-feature mean, 2.5th, 97.5th
    percentiles of the bootstrap distribution.
    """
    n = len(y)
    rng = np.random.default_rng(42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    boot_coefs = []
    boot_intercepts = []
    for _ in range(n_resamples):
        # Stratified bootstrap: sample class 0 and class 1 separately
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        boot_idx = np.concatenate(
            [
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ]
        )
        rng.shuffle(boot_idx)
        X_b, y_b = X_scaled[boot_idx], y[boot_idx]

        lr = LogisticRegression(C=C, max_iter=1000, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(X_b, y_b)

        boot_coefs.append(lr.coef_[0].tolist())
        boot_intercepts.append(float(lr.intercept_[0]))

    boot_coefs = np.array(boot_coefs)  # shape: (n_resamples, n_features)
    boot_intercepts = np.array(boot_intercepts)

    ci_results = {}
    for i, feat in enumerate(FEATURES):
        vals = boot_coefs[:, i]
        ci_results[feat] = {
            "mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
            "std": float(vals.std()),
        }

    ci_results["intercept"] = {
        "mean": float(boot_intercepts.mean()),
        "ci_lo": float(np.percentile(boot_intercepts, 2.5)),
        "ci_hi": float(np.percentile(boot_intercepts, 97.5)),
        "std": float(boot_intercepts.std()),
    }

    return ci_results


def compute_calibration(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, n_folds: int, n_bins: int = 10) -> dict:
    """Reliability diagram data via CV predictions."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frac_pos, mean_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy="uniform")

    # Expected calibration error
    bin_size = 1.0 / n_bins
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    return {
        "fraction_positive": [float(v) for v in frac_pos],
        "mean_predicted": [float(v) for v in mean_pred],
        "ece": ece,
        "n_bins": n_bins,
    }


def fit_final_model(X: np.ndarray, y: np.ndarray, C: float) -> tuple[Pipeline, dict]:
    """Fit final logistic regression on full dataset. Return pipeline + params."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=1000, random_state=0)),
        ]
    )
    pipeline.fit(X, y)

    scaler = pipeline.named_steps["scaler"]
    lr = pipeline.named_steps["lr"]

    params = {
        "feature_means": scaler.mean_.tolist(),
        "feature_stds": scaler.scale_.tolist(),
        "coef_standardized": lr.coef_[0].tolist(),  # coefficients on standardized scale
        "intercept_standardized": float(lr.intercept_[0]),
        # Coefficients on original (unstandardized) scale:
        #   coef_orig[i] = coef_std[i] / feature_std[i]
        #   intercept_orig = intercept_std - sum(coef_std * mean / std)
        "coef_original": (lr.coef_[0] / scaler.scale_).tolist(),
        "intercept_original": float(lr.intercept_[0] - np.dot(lr.coef_[0], scaler.mean_ / scaler.scale_)),
        "regularization_C": C,
    }
    return pipeline, params


def find_best_C(X: np.ndarray, y: np.ndarray, n_folds: int) -> float:
    """Grid search over C values; pick highest AUC on stratified CV."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    candidates = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_C, best_auc = candidates[0], -1.0

    for C in candidates:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C, max_iter=1000, random_state=0)),
            ]
        )
        y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, y_prob)
        if auc > best_auc:
            best_auc, best_C = auc, C

    print(f"  Best C={best_C} (CV AUC={best_auc:.4f})")
    return best_C


def print_model_card(model_json: dict) -> None:
    """Print a concise model card to stdout."""
    cv = model_json["cross_validation"]
    boot = model_json["bootstrap_cis"]
    params = model_json["model_params"]

    print("\n" + "=" * 68)
    print("  EMPIRICAL COOPERATION RATE MODEL — MODEL CARD")
    print("=" * 68)
    print(f"  Dataset     : ESS Round 11, Austria (N={cv['n_samples']})")
    print(f"  Target      : volunteered (base rate={cv['base_rate']:.1%})")
    print(f"  Model       : Logistic regression (L2, C={params['regularization_C']})")
    print(f"  Validation  : {cv['n_folds']}-fold stratified CV")
    print()
    print(f"  CV AUC      : {cv['auc_mean']:.4f}  ±{cv['auc_std']:.4f}  (per-fold)")
    print(
        f"  Brier score : {cv['brier_score']:.4f}  (lower = better; null={cv['base_rate'] * (1 - cv['base_rate']):.4f})"
    )
    print(f"  ECE         : {model_json['calibration']['ece']:.4f}  (expected calibration error)")
    print()
    print("  Feature coefficients (standardized, 95% bootstrap CI):")
    print(f"  {'Feature':<25} {'Coef':>8}  {'95% CI':>20}  {'Sig?':>6}")
    print("  " + "-" * 65)
    for feat in FEATURES:
        b = boot[feat]
        sig = "YES" if not (b["ci_lo"] < 0 < b["ci_hi"]) else "no"
        ci_str = f"[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]"
        print(f"  {feat:<25} {b['mean']:>+8.3f}  {ci_str:>20}  {sig:>6}")
    print()
    print("  Interpretation:")
    print("    Positive coef → predictor increases P(volunteer/cooperate)")
    print("    Negative coef → predictor decreases P(volunteer/cooperate)")
    print()
    print("  Known limitation: Austria-only data; cross-country generalization")
    print("  is untested. Cross-cultural cooperation baselines may differ.")
    print()
    print("  Replaces heuristic: 0.2 + 0.6 * trust * (1 - risk)")
    print("  Now uses: logistic(β·x) with empirically estimated β")
    print("=" * 68)


def main(ess_path: Path, n_bootstrap: int, cv_folds: int, output_path: Path) -> None:
    print(f"Loading ESS data from {ess_path} ...")
    X, y, df = load_and_prepare(ess_path)
    print(f"  N={len(y)}, positive (volunteered)={y.sum()} ({y.mean():.1%})")
    print(f"  Features: {FEATURES}")

    print(f"\nSearching for best regularization strength (C) via {cv_folds}-fold CV ...")
    best_C = find_best_C(X, y, cv_folds)

    # Build final pipeline with best C for all subsequent steps
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best_C, max_iter=1000, random_state=0)),
        ]
    )

    print(f"\nRunning {cv_folds}-fold stratified cross-validation ...")
    cv_metrics = cross_validate(final_pipeline, X, y, cv_folds)
    print(f"  AUC={cv_metrics['auc_mean']:.4f} ± {cv_metrics['auc_std']:.4f}")
    print(f"  Brier={cv_metrics['brier_score']:.4f}")

    print("\nComputing calibration curve ...")
    calib = compute_calibration(final_pipeline, X, y, cv_folds)
    print(f"  ECE={calib['ece']:.4f}")

    print(f"\nBootstrapping {n_bootstrap} resamples for 95% CIs ...")
    boot_cis = bootstrap_coefficients(X, y, n_bootstrap, best_C)

    print("\nFitting final model on full dataset ...")
    final_pipeline, model_params = fit_final_model(X, y, best_C)

    # Assemble output
    model_json = {
        "meta": {
            "description": (
                "Logistic regression cooperate/not model fitted on ESS Round 11 "
                "Austrian volunteering data. Replaces the heuristic formula "
                "'0.2 + 0.6 * trust_people * (1 - risk_tolerance)' in "
                "metrics/persona_decay.py with empirically estimated coefficients."
            ),
            "ess_source": str(ess_path),
            "ess_country": "AT",
            "ess_round": 11,
            "target_variable": TARGET,
            "target_encoding": "1=volunteered (cooperate), 0=did not volunteer",
            "features": FEATURES,
            "feature_ess_notes": {
                "trust_people": "ESS ppltrst: most people can be trusted (0-1)",
                "trust_fairness": "ESS pplfair: most people try to be fair (0-1)",
                "trust_helpfulness": "ESS pplhlp: most people try to be helpful (0-1)",
                "risk_taking": "ESS risk: willing to take risks (0-1); note: BGF profiles use risk_tolerance which is this variable",
                "social_meeting_freq": "ESS sclmeet: how often meet socially (0-1)",
                "social_activity": "ESS sclact: take part in social activities (0-1)",
                "reduce_inequality": "ESS gincdif: government should reduce income differences (0-1)",
            },
            "model_type": "LogisticRegression(L2)",
            "n_bootstrap": n_bootstrap,
            "cv_folds": cv_folds,
            "known_limitations": [
                "Austria-only data (N=866); cross-country generalization untested",
                "Volunteering is a cooperation proxy, not a direct cooperation measure",
                "ESS does not include experimental trust-game outcomes",
                "Fitted on attitude/behavior items from 2022-2023 data collection",
            ],
        },
        "model_params": model_params,
        "cross_validation": cv_metrics,
        "bootstrap_cis": boot_cis,
        "calibration": calib,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(model_json, f, indent=2)

    print_model_card(model_json)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ess-path", type=Path, default=Path("data/ess_clean.parquet"))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=10)
    args = parser.parse_args()

    main(
        ess_path=args.ess_path,
        n_bootstrap=args.n_bootstrap,
        cv_folds=args.cv_folds,
        output_path=args.output,
    )
