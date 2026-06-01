"""Synthetic data utility benchmark — Train on Synthetic, Test on Real (TSTR).

The TSTR benchmark (Jordon et al. 2022; Synthetic Data Vault) is the canonical
proof that synthetic data can *replace* real data for downstream modelling.

Protocol
--------
1. Generate a synthetic agent population using BGF (the ``build_synthetic_dataset``
   function converts agent profiles into feature vectors).
2. Train a downstream predictor on synthetic features + labels.
3. Test the predictor on *real* ESS data (held-out from grounding).
4. Compare TSTR accuracy to TRTR (Train Real, Test Real) accuracy.
5. Compute the **Utility Gap**: TRTR − TSTR (target: < 0.05, i.e. within 5 pp).

Downstream task
---------------
Binary classification: predict ``high_cooperation_propensity`` from demographic
attributes.  The label is derived as::

    high_coop = 1  iff  trust_people >= 0.5

Features used: age (normalised), income_decile (normalised), education_years
(normalised), trust_people (for label only), social_activity (normalised).

This is intentionally simple so that it can run CPU-only without GPU.
The task validates whether the *joint distribution* of demographic features
in synthetic BGF data is faithful enough to train a working predictor.

Estimator
---------
Logistic regression (scikit-learn) with L2 regularisation (C=1.0).
AUC-ROC is the primary metric; accuracy is secondary.

Usage
-----
>>> from metrics.synthetic_utility import (
...     build_synthetic_dataset,
...     tstr_benchmark,
...     utility_report,
... )
>>> synthetic_X, synthetic_y = build_synthetic_dataset(synthetic_profiles)
>>> real_X, real_y = load_real_ess_dataset("data/ess_clean.parquet")
>>> result = tstr_benchmark(synthetic_X, synthetic_y, real_X, real_y)
>>> print(utility_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Feature/label extraction ─────────────────────────────────────────────────

#: Feature columns (in order) extracted from agent profiles / ESS rows.
FEATURE_COLUMNS = [
    "age_norm",
    "income_norm",
    "education_norm",
    "social_activity_norm",
]

#: Label: trust_people >= 0.5 → 1 (high cooperation propensity).
TRUST_THRESHOLD = 0.5

#: Maximum acceptable utility gap for the "replace real data" claim.
MAX_UTILITY_GAP = 0.05


# ── Profile → feature vector ─────────────────────────────────────────────────


def _safe_norm(val, lo: float, hi: float, default: float = 0.5) -> float:
    """Normalise val from [lo, hi] to [0, 1]; return default if val is None."""
    if val is None:
        return default
    try:
        v = float(val)
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else default
    except (TypeError, ValueError):
        return default


def profile_to_feature_vector(profile) -> list[float]:
    """Convert an AgentProfile to a FEATURE_COLUMNS-ordered feature vector.

    Handles both AgentProfile objects (with attribute access) and plain dicts.

    Args:
        profile: AgentProfile instance or dict with demographic fields.

    Returns:
        Length-4 float list matching FEATURE_COLUMNS order.
    """
    if isinstance(profile, dict):
        age = profile.get("age", 35)
        income_decile = profile.get("income_decile", 5)
        education = profile.get("education_years", 10)
        social = profile.get("social_activity", 0.5)
    else:
        age = getattr(profile, "age", 35)
        # income field may be raw value or normalised decile
        income_raw = getattr(profile, "income", None)
        income_decile = getattr(profile, "income_decile", (income_raw / 1000.0) if income_raw else 5)
        education = getattr(profile, "education_years", getattr(profile, "education", 10))
        social = getattr(profile, "social_activity", 0.5)

    return [
        _safe_norm(age, 15, 90),
        _safe_norm(income_decile, 1, 10),
        _safe_norm(education, 0, 25),
        _safe_norm(social, 0.0, 1.0) if isinstance(social, float) else _safe_norm(social, 1, 7),
    ]


def profile_to_label(profile) -> int:
    """Extract binary cooperation-propensity label from a profile.

    Returns 1 if trust_people >= TRUST_THRESHOLD, else 0.
    """
    if isinstance(profile, dict):
        trust = profile.get("trust_people", 0.5)
    else:
        trust = getattr(profile, "trust_people", 0.5)
    return int(float(trust or 0.5) >= TRUST_THRESHOLD)


def build_synthetic_dataset(
    profiles: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of AgentProfile objects into (X, y) arrays for TSTR.

    Args:
        profiles: List of AgentProfile instances (or dicts with same fields).

    Returns:
        X: float32 array of shape (n_samples, n_features).
        y: int32 label array of shape (n_samples,).
    """
    X = np.array([profile_to_feature_vector(p) for p in profiles], dtype=np.float32)
    y = np.array([profile_to_label(p) for p in profiles], dtype=np.int32)
    return X, y


def load_real_ess_dataset(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load real ESS data from Parquet and convert to (X, y) arrays.

    Args:
        parquet_path: Path to ``data/ess_clean.parquet``.

    Returns:
        X, y arrays (same format as build_synthetic_dataset).

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ImportError: If pandas or pyarrow is not installed.
    """
    from pathlib import Path

    import pandas as pd

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ESS clean dataset not found at {path}. Run scripts/download_ess_data.py to generate it."
        )

    df = pd.read_parquet(path)

    rows = []
    labels = []
    for _, row in df.iterrows():
        feat = profile_to_feature_vector(row.to_dict())
        rows.append(feat)
        labels.append(int(float(row.get("trust_people", 0.5)) >= TRUST_THRESHOLD))

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


# ── Classifier ───────────────────────────────────────────────────────────────


def _fit_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a logistic regression on train data, return (predicted_labels, predicted_probs)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for TSTR benchmark. Install with: pip install scikit-learn"
        ) from exc

    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X)
    test_X_s = scaler.transform(test_X)

    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(train_X_s, train_y)

    pred_labels = clf.predict(test_X_s)
    pred_probs = clf.predict_proba(test_X_s)[:, 1]
    return pred_labels, pred_probs


def _compute_metrics(
    y_true: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and AUC-ROC."""
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn required") from exc

    n_classes = len(np.unique(y_true))
    acc = float(accuracy_score(y_true, pred_labels))
    if n_classes < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, pred_probs))
    return {"accuracy": round(acc, 4), "auc": round(auc, 4)}


# ── TSTR result ───────────────────────────────────────────────────────────────


@dataclass
class TSTRResult:
    """Result of the TSTR benchmark.

    Attributes:
        tstr_accuracy: Accuracy when trained on synthetic, tested on real.
        tstr_auc: AUC-ROC for TSTR.
        trtr_accuracy: Accuracy when trained on real, tested on real (baseline).
        trtr_auc: AUC-ROC for TRTR.
        utility_gap_accuracy: trtr_accuracy - tstr_accuracy (lower = better).
        utility_gap_auc: trtr_auc - tstr_auc.
        passes_utility_threshold: True iff utility_gap_accuracy ≤ MAX_UTILITY_GAP.
        n_synthetic_train: Number of synthetic training samples.
        n_real_train: Number of real training samples (for TRTR).
        n_real_test: Number of real test samples.
        synthetic_label_balance: Fraction of positive labels in synthetic data.
        real_label_balance: Fraction of positive labels in real data.
    """

    tstr_accuracy: float
    tstr_auc: float
    trtr_accuracy: float
    trtr_auc: float
    utility_gap_accuracy: float
    utility_gap_auc: float
    passes_utility_threshold: bool
    n_synthetic_train: int
    n_real_train: int
    n_real_test: int
    synthetic_label_balance: float
    real_label_balance: float


# ── Core TSTR function ────────────────────────────────────────────────────────


def tstr_benchmark(
    synthetic_X: np.ndarray,
    synthetic_y: np.ndarray,
    real_X: np.ndarray,
    real_y: np.ndarray,
    test_fraction: float = 0.30,
    random_state: int = 42,
) -> TSTRResult:
    """Run the TSTR (Train Synthetic, Test Real) benchmark.

    1. Split real data into train (TRTR) and test sets.
    2. Train LogisticRegression on synthetic data → test on real test set (TSTR).
    3. Train LogisticRegression on real train data → test on same real test set (TRTR).
    4. Compute utility gap (TRTR - TSTR). Small gap = synthetic is a valid substitute.

    Args:
        synthetic_X: Feature matrix for synthetic training data.
        synthetic_y: Labels for synthetic data.
        real_X: Feature matrix for real data.
        real_y: Labels for real data.
        test_fraction: Fraction of real data held out for testing.
        random_state: Random seed for train/test split.

    Returns:
        TSTRResult with all metrics.

    Raises:
        ValueError: If either dataset has fewer than 10 samples.
    """
    if len(synthetic_X) < 10:
        raise ValueError(f"Synthetic dataset too small: {len(synthetic_X)} samples (min 10).")
    if len(real_X) < 10:
        raise ValueError(f"Real dataset too small: {len(real_X)} samples (min 10).")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("scikit-learn required") from exc

    # Split real into train/test
    real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
        real_X,
        real_y,
        test_size=test_fraction,
        random_state=random_state,
        stratify=real_y if len(np.unique(real_y)) > 1 else None,
    )

    # TSTR: train on synthetic, test on real_test
    tstr_pred, tstr_probs = _fit_predict(synthetic_X, synthetic_y, real_X_test)
    tstr_metrics = _compute_metrics(real_y_test, tstr_pred, tstr_probs)

    # TRTR: train on real_train, test on real_test
    trtr_pred, trtr_probs = _fit_predict(real_X_train, real_y_train, real_X_test)
    trtr_metrics = _compute_metrics(real_y_test, trtr_pred, trtr_probs)

    gap_acc = round(trtr_metrics["accuracy"] - tstr_metrics["accuracy"], 4)
    gap_auc = round(trtr_metrics["auc"] - tstr_metrics["auc"], 4)

    synth_balance = float(np.mean(synthetic_y))
    real_balance = float(np.mean(real_y))

    return TSTRResult(
        tstr_accuracy=tstr_metrics["accuracy"],
        tstr_auc=tstr_metrics["auc"],
        trtr_accuracy=trtr_metrics["accuracy"],
        trtr_auc=trtr_metrics["auc"],
        utility_gap_accuracy=gap_acc,
        utility_gap_auc=gap_auc,
        passes_utility_threshold=gap_acc <= MAX_UTILITY_GAP,
        n_synthetic_train=len(synthetic_X),
        n_real_train=len(real_X_train),
        n_real_test=len(real_X_test),
        synthetic_label_balance=round(synth_balance, 4),
        real_label_balance=round(real_balance, 4),
    )


# ── Report ────────────────────────────────────────────────────────────────────


def utility_report(result: TSTRResult) -> str:
    """Return a human-readable TSTR benchmark summary."""
    gap_status = "PASS ✓" if result.passes_utility_threshold else "FAIL ✗"
    lines = [
        "=" * 60,
        "  Synthetic Data Utility Benchmark (TSTR vs TRTR)",
        "=" * 60,
        "",
        "  Dataset sizes",
        f"    Synthetic train:  {result.n_synthetic_train:>6}",
        f"    Real train:       {result.n_real_train:>6}",
        f"    Real test:        {result.n_real_test:>6}",
        "",
        "  Label balance (fraction positive)",
        f"    Synthetic:        {result.synthetic_label_balance:.4f}",
        f"    Real:             {result.real_label_balance:.4f}",
        "",
        f"  {'Condition':<20} {'Accuracy':>10} {'AUC-ROC':>10}",
        f"  {'-' * 42}",
        f"  {'TRTR (train real)':<20} {result.trtr_accuracy:>10.4f} {result.trtr_auc:>10.4f}",
        f"  {'TSTR (train synth)':<20} {result.tstr_accuracy:>10.4f} {result.tstr_auc:>10.4f}",
        f"  {'-' * 42}",
        f"  {'Utility gap':<20} {result.utility_gap_accuracy:>10.4f} {result.utility_gap_auc:>10.4f}",
        "",
        f"  Utility threshold (≤ {MAX_UTILITY_GAP}):  {gap_status}",
        "",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)
    return report
