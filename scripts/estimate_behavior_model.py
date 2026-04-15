"""
Estimate a behavioral model from cleaned ESS data.

Uses the full ESS feature set (trust, risk, social activity, satisfaction, etc.)
to train a multinomial logistic regression for behavioral action prediction.

Usage:
    python scripts/estimate_behavior_model.py
    python scripts/estimate_behavior_model.py --input data/ess_clean.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ── Feature sets ─────────────────────────────────────────────────────────────

# ESS-derived features for behavioral prediction
ESS_FEATURES = [
    "age",
    "gender",
    "trust_people",
    "trust_fairness",
    "trust_helpfulness",
    "trust_parliament",
    "trust_legal",
    "trust_police",
    "left_right",
    "life_satisfaction",
    "happiness",
    "risk_taking",
    "competitiveness",
    "leadership_preference",
    "social_meeting_freq",
    "self_rated_health",
    "satisfaction_economy",
    "immigration_same_ethnicity",
    "reduce_inequality",
]


def generate_behavioral_target(df: pd.DataFrame) -> pd.Series:
    """
    Generate behavioral proxy targets from ESS survey responses.

    Richer classification than the original age/gender heuristic:
      - High trust + high social → cooperate
      - High risk + high competitiveness → work
      - Low risk + high satisfaction → save
    """

    def classify(row):
        trust = row.get("trust_people", 0.5)
        social = row.get("social_meeting_freq", 0.5)
        risk = row.get("risk_taking", 0.5)
        satisfaction = row.get("life_satisfaction", 0.5)
        competitiveness = row.get("competitiveness", 0.5)

        # Use NaN-safe defaults
        for var_name in ["trust", "social", "risk", "satisfaction", "competitiveness"]:
            v = locals()[var_name]
            if pd.isna(v):
                locals()[var_name] = 0.5

        score_cooperate = 0.5 * trust + 0.4 * social + 0.1 * satisfaction
        score_work = 0.4 * risk + 0.3 * competitiveness + 0.15 * (1 - satisfaction) + 0.15
        score_save = 0.4 * (1 - risk) + 0.3 * satisfaction + 0.2 * (1 - trust)

        scores = {"cooperate": score_cooperate, "work": score_work, "save": score_save}
        return max(scores, key=scores.get)

    return df.apply(classify, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Estimate behavioral model from ESS data.")
    parser.add_argument("--input", default="data/ess_clean.parquet", help="Cleaned ESS data")
    parser.add_argument("--output", default="data/behavior/behavior_coefficients.json", help="Output path")
    args = parser.parse_args()

    print("=" * 50)
    print("Behavioral Model Estimation — BGF")
    print("=" * 50)

    # Load data
    df = pd.read_parquet(args.input)
    print(f"Loaded: {df.shape}")

    # Select features that exist in the dataset
    available_features = [f for f in ESS_FEATURES if f in df.columns]
    missing_features = [f for f in ESS_FEATURES if f not in df.columns]

    if missing_features:
        print(f"Warning: {len(missing_features)} features not found: {missing_features}")

    print(f"Using {len(available_features)} features: {available_features}")

    # Generate behavioral target
    df["action"] = generate_behavioral_target(df)
    print("\nAction distribution:")
    for action, count in df["action"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")

    # Prepare features
    X = df[available_features].copy()
    y = df["action"]

    # Fill NaN with column median
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Export coefficients
    coefficients = {
        "features": available_features,
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_samples": int(len(df)),
        "n_features": len(available_features),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coefficients, f, indent=2)

    print(f"\nModel saved to: {output_path}")
    print("Model estimated successfully")


if __name__ == "__main__":
    main()
