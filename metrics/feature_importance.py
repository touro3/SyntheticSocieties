"""ESS Feature Importance Analysis — Phase 28.3.

Answers "which ESS profile dimensions drive cooperation?" via logistic
regression on per-round agent decisions.  Produces coefficient rankings,
odds ratios, and a per-dimension ablation table.

Usage
-----
    from metrics.feature_importance import (
        build_feature_matrix,
        run_logistic_regression,
        compute_ablation_table,
        FeatureImportanceResult,
    )
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# scikit-learn is in requirements.txt (scikit-learn==1.5.2)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Feature definitions ───────────────────────────────────────────────────────

#: Ordered list of ESS profile attribute names used as predictors.
ESS_FEATURE_NAMES: list[str] = [
    "trust_people",
    "trust_institutions",
    "risk_tolerance",
    "social_activity",
    "life_satisfaction",
    "happiness",
    "competitiveness",
    "leadership_preference",
    "health_status",
    "religiosity",
    "political_orientation",
    "immigration_attitude",
]

#: Minimal profile — only interpersonal trust and risk.
PROFILE_MINIMAL: list[str] = ["trust_people", "risk_tolerance"]

#: Medium profile — trust, risk, income proxy, age proxy.
PROFILE_MEDIUM: list[str] = [
    "trust_people",
    "risk_tolerance",
    "social_activity",
    "life_satisfaction",
]

#: Full profile — all ESS dimensions.
PROFILE_FULL: list[str] = ESS_FEATURE_NAMES


# ── Data structures ───────────────────────────────────────────────────────────


class AgentRoundRecord(NamedTuple):
    """One observation: agent profile attributes + binary cooperation outcome."""

    trust_people: float
    trust_institutions: float
    risk_tolerance: float
    social_activity: float
    life_satisfaction: float
    happiness: float
    competitiveness: float
    leadership_preference: float
    health_status: float
    religiosity: float
    political_orientation: float
    immigration_attitude: float
    cooperated: int  # 1 if action == "cooperate", else 0


@dataclass
class FeatureCoefficient:
    """Logistic regression coefficient for a single ESS feature."""

    feature: str
    coefficient: float
    odds_ratio: float
    abs_rank: int  # 1 = most important


@dataclass
class FeatureImportanceResult:
    """Full output of a feature importance analysis run."""

    coefficients: list[FeatureCoefficient]
    intercept: float
    n_observations: int
    n_cooperate: int
    cooperation_rate: float
    feature_names: list[str]
    train_accuracy: float
    # Ablation results: profile_level → mean cooperation rate
    ablation_table: dict[str, float] = field(default_factory=dict)

    def top_features(self, n: int = 5) -> list[FeatureCoefficient]:
        """Return the top-n features by absolute coefficient magnitude."""
        return sorted(self.coefficients, key=lambda c: abs(c.coefficient), reverse=True)[:n]

    def to_table_rows(self) -> list[dict]:
        """Render coefficients as a list of dicts suitable for printing/CSV."""
        return [
            {
                "rank": fc.abs_rank,
                "feature": fc.feature,
                "coefficient": round(fc.coefficient, 4),
                "odds_ratio": round(fc.odds_ratio, 4),
            }
            for fc in sorted(self.coefficients, key=lambda c: c.abs_rank)
        ]


# ── Core functions ────────────────────────────────────────────────────────────


def build_feature_matrix(
    records: Iterable[AgentRoundRecord],
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) from a sequence of AgentRoundRecord observations.

    Args:
        records: Iterable of per-round agent observations.
        feature_names: Subset of :data:`ESS_FEATURE_NAMES` to include.
            Defaults to all features.

    Returns:
        Tuple (X, y) where X has shape (n, len(features)) and y is binary
        cooperation indicator of shape (n,).

    Raises:
        ValueError: If records is empty or feature_names contains unknown names.
    """
    names = feature_names if feature_names is not None else ESS_FEATURE_NAMES

    unknown = set(names) - set(ESS_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}")

    rows_x: list[list[float]] = []
    rows_y: list[int] = []

    for rec in records:
        row = [getattr(rec, feat) for feat in names]
        rows_x.append(row)
        rows_y.append(rec.cooperated)

    if not rows_x:
        raise ValueError("No records provided — cannot build feature matrix.")

    X = np.array(rows_x, dtype=float)
    y = np.array(rows_y, dtype=int)
    return X, y


def run_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    C: float = 1.0,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """Fit a logistic regression and return a FeatureImportanceResult.

    Features are z-scored before fitting so coefficients are comparable
    across dimensions with different scales.

    Args:
        X: Feature matrix (n_samples × n_features).
        y: Binary cooperation labels (n_samples,).
        feature_names: Names corresponding to columns of X.
        C: Inverse of L2 regularization strength (sklearn convention).
        random_state: For reproducibility.

    Returns:
        :class:`FeatureImportanceResult` with ranked coefficients.

    Raises:
        ValueError: If X and y have incompatible shapes.
    """
    if X.shape[0] != len(y):
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {len(y)} elements."
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X.shape[1]} columns but {len(feature_names)} feature names given."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=C, max_iter=500, random_state=random_state, solver="lbfgs")
    clf.fit(X_scaled, y)

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)
    train_acc = float(clf.score(X_scaled, y))

    # Rank by absolute value
    abs_order = np.argsort(np.abs(coefs))[::-1]
    rank_map = {idx: rank + 1 for rank, idx in enumerate(abs_order)}

    coefficients = [
        FeatureCoefficient(
            feature=feature_names[i],
            coefficient=float(coefs[i]),
            odds_ratio=float(odds_ratios[i]),
            abs_rank=rank_map[i],
        )
        for i in range(len(feature_names))
    ]

    n = int(len(y))
    n_coop = int(np.sum(y))

    return FeatureImportanceResult(
        coefficients=coefficients,
        intercept=float(clf.intercept_[0]),
        n_observations=n,
        n_cooperate=n_coop,
        cooperation_rate=n_coop / n if n > 0 else 0.0,
        feature_names=feature_names,
        train_accuracy=train_acc,
    )


def compute_ablation_table(
    records: list[AgentRoundRecord],
    profile_levels: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Fit logistic regression at each profile depth and return train accuracy.

    This shows the monotonic accuracy improvement as more ESS dimensions are
    included — the "profile richness vs. cooperation prediction" table.

    Args:
        records: Per-round agent observations.
        profile_levels: Mapping of level name → feature list. Defaults to
            ``{minimal, medium, full}``.

    Returns:
        Dict mapping profile level name → train accuracy of logistic regression.
    """
    levels = profile_levels or {
        "minimal": PROFILE_MINIMAL,
        "medium": PROFILE_MEDIUM,
        "full": PROFILE_FULL,
    }

    ablation: dict[str, float] = {}
    for level_name, features in levels.items():
        X, y = build_feature_matrix(records, feature_names=features)
        result = run_logistic_regression(X, y, feature_names=features)
        ablation[level_name] = result.train_accuracy

    return ablation


def records_from_profile_actions(
    profiles: list,
    action_sequences: list[list[str]],
) -> list[AgentRoundRecord]:
    """Convert BGF profiles + action sequences to AgentRoundRecord observations.

    Each profile contributes one record per round (one per action in its
    action_sequence), using the profile's ESS attributes.

    Args:
        profiles: List of AgentProfile objects with ESS attributes.
        action_sequences: Parallel list of action lists (one per agent).
            Each inner list contains action strings ("work", "save", "cooperate").

    Returns:
        Flat list of AgentRoundRecord — one per (agent, round) pair.

    Raises:
        ValueError: If profiles and action_sequences have different lengths.
    """
    if len(profiles) != len(action_sequences):
        raise ValueError(
            f"Got {len(profiles)} profiles but {len(action_sequences)} action sequences."
        )

    records: list[AgentRoundRecord] = []
    for profile, actions in zip(profiles, action_sequences):
        for action in actions:
            rec = AgentRoundRecord(
                trust_people=float(profile.trust_people or 0.5),
                trust_institutions=float(profile.trust_institutions or 0.5),
                risk_tolerance=float(profile.risk_tolerance or 0.5),
                social_activity=float(profile.social_activity or 0.5),
                life_satisfaction=float(profile.life_satisfaction or 0.5),
                happiness=float(profile.happiness or 0.5),
                competitiveness=float(profile.competitiveness or 0.5),
                leadership_preference=float(profile.leadership_preference or 0.5),
                health_status=float(profile.health_status or 0.5),
                religiosity=float(profile.religiosity or 0.5),
                political_orientation=float(profile.political_orientation or 0.5),
                immigration_attitude=float(profile.immigration_attitude or 0.5),
                cooperated=1 if action == "cooperate" else 0,
            )
            records.append(rec)

    return records
