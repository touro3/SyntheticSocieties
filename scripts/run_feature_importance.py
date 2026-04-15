"""Feature Importance Analysis — Phase 28.3.

Generates the ESS feature importance figure and ablation table by:
  1. Running a short simulation with the rule-based ESS policy (no GPU needed).
  2. Extracting per-round (profile, action) pairs.
  3. Fitting logistic regression: cooperate ~ ESS profile attributes.
  4. Saving results to analysis/tables/feature_importance.json.

Usage
-----
    python scripts/run_feature_importance.py
    python scripts/run_feature_importance.py --n-agents 200 --n-rounds 20
    python scripts/run_feature_importance.py --plot-only  # re-plot from saved json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.profile import AgentProfile
from metrics.feature_importance import (
    PROFILE_FULL,
    AgentRoundRecord,
    FeatureImportanceResult,
    build_feature_matrix,
    compute_ablation_table,
    run_logistic_regression,
)

# ── Synthetic data generation (no GPU, no ESS file required) ─────────────────


def _synthetic_profiles(n: int, seed: int = 42) -> list[AgentProfile]:
    """Generate N synthetic AgentProfiles with random ESS attributes."""
    rng = np.random.default_rng(seed)

    profiles = []
    for i in range(n):
        trust = float(rng.beta(2, 2))  # centered ~0.5, realistic spread
        risk = float(rng.beta(2, 3))  # slightly risk-averse on average
        social = float(rng.beta(2, 2))
        life_sat = float(rng.beta(3, 2))  # slightly positive on average
        happiness = float(rng.beta(3, 2))
        competitiveness = float(rng.beta(2, 3))
        leadership = float(rng.beta(2, 3))
        health = float(rng.beta(3, 2))
        religiosity = float(rng.beta(1.5, 3))
        political = float(rng.uniform(0.0, 1.0))
        immigration = float(rng.beta(2, 2))
        trust_inst = float(rng.beta(2, 2))

        income = float(rng.uniform(10_000, 80_000))
        age = int(rng.integers(18, 75))

        profiles.append(
            AgentProfile(
                agent_id=f"agent_{i:04d}",
                age=age,
                income=income,
                education="secondary",
                occupation="employed",
                location="AT",
                political_preference="moderate",
                risk_tolerance=risk,
                social_class="middle",
                gender=int(rng.integers(1, 3)),
                country="AT",
                education_level=int(rng.integers(1, 8)),
                income_decile=int(rng.integers(1, 11)),
                trust_people=trust,
                trust_institutions=trust_inst,
                political_orientation=political,
                life_satisfaction=life_sat,
                happiness=happiness,
                immigration_attitude=immigration,
                social_activity=social,
                competitiveness=competitiveness,
                leadership_preference=leadership,
                health_status=health,
                religiosity=religiosity,
            )
        )
    return profiles


def _simulate_actions(
    profiles: list[AgentProfile],
    n_rounds: int,
    seed: int = 42,
) -> list[AgentRoundRecord]:
    """Simulate rule-based actions and return AgentRoundRecord observations.

    The action probability mirrors the RuleBasedESSPolicy formula so the
    feature importance result is consistent with Condition D behavior.
    """
    import hashlib
    import struct

    def _hash_uniform(agent_id: str, round_id: int) -> float:
        key = f"{agent_id}:{round_id}".encode()
        digest = hashlib.sha256(key).digest()
        uint32 = struct.unpack(">I", digest[:4])[0]
        return uint32 / 4_294_967_296.0

    records: list[AgentRoundRecord] = []
    for profile in profiles:
        for r in range(n_rounds):
            trust = profile.trust_people or 0.5
            risk = profile.risk_tolerance or 0.5
            social = profile.social_activity or 0.5
            coop_prob = max(0.05, min(0.90, 0.2 + 0.5 * trust * (1 - risk) + 0.15 * social))

            h = _hash_uniform(profile.agent_id, r)
            cooperated = 1 if h < coop_prob else 0

            records.append(
                AgentRoundRecord(
                    trust_people=profile.trust_people or 0.5,
                    trust_institutions=profile.trust_institutions or 0.5,
                    risk_tolerance=profile.risk_tolerance or 0.5,
                    social_activity=profile.social_activity or 0.5,
                    life_satisfaction=profile.life_satisfaction or 0.5,
                    happiness=profile.happiness or 0.5,
                    competitiveness=profile.competitiveness or 0.5,
                    leadership_preference=profile.leadership_preference or 0.5,
                    health_status=profile.health_status or 0.5,
                    religiosity=profile.religiosity or 0.5,
                    political_orientation=profile.political_orientation or 0.5,
                    immigration_attitude=profile.immigration_attitude or 0.5,
                    cooperated=cooperated,
                )
            )
    return records


# ── Main ──────────────────────────────────────────────────────────────────────


def run(n_agents: int = 300, n_rounds: int = 30, seed: int = 42) -> FeatureImportanceResult:
    print(f"[feature_importance] Generating {n_agents} profiles × {n_rounds} rounds…")
    profiles = _synthetic_profiles(n_agents, seed=seed)
    records = _simulate_actions(profiles, n_rounds, seed=seed)
    print(f"[feature_importance] {len(records):,} observations. Fitting logistic regression…")

    X, y = build_feature_matrix(records, feature_names=PROFILE_FULL)
    result = run_logistic_regression(X, y, feature_names=PROFILE_FULL)

    print(f"[feature_importance] Train accuracy: {result.train_accuracy:.4f}")
    print(f"[feature_importance] Cooperation rate: {result.cooperation_rate:.4f}")
    print("[feature_importance] Top 5 features by |coefficient|:")
    for fc in result.top_features(5):
        sign = "↑" if fc.coefficient > 0 else "↓"
        print(f"  #{fc.abs_rank:2d} {fc.feature:30s} coef={fc.coefficient:+.4f}  OR={fc.odds_ratio:.3f} {sign}")

    # Ablation
    print("[feature_importance] Running profile-depth ablation…")
    ablation = compute_ablation_table(records)
    result.ablation_table = ablation
    print("[feature_importance] Ablation (train accuracy):")
    for level, acc in ablation.items():
        print(f"  {level:10s}: {acc:.4f}")

    return result


def save_results(result: FeatureImportanceResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_importance.json"

    payload = {
        "n_observations": result.n_observations,
        "n_cooperate": result.n_cooperate,
        "cooperation_rate": result.cooperation_rate,
        "train_accuracy": result.train_accuracy,
        "coefficients": result.to_table_rows(),
        "ablation_table": result.ablation_table,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[feature_importance] Results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESS feature importance analysis")
    parser.add_argument("--n-agents", type=int, default=300)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation; re-plot from existing JSON.")
    args = parser.parse_args()

    tables_dir = Path("analysis/tables")

    if not args.plot_only:
        result = run(n_agents=args.n_agents, n_rounds=args.n_rounds, seed=args.seed)
        save_results(result, tables_dir)

    # Always attempt to plot if the JSON exists
    json_path = tables_dir / "feature_importance.json"
    if json_path.exists():
        print("[feature_importance] Generating figure…")
        import subprocess

        subprocess.run(
            ["python", "scripts/plot_feature_importance.py", "--input", str(json_path)],
            check=False,
        )
    else:
        print(f"[feature_importance] JSON not found at {json_path}; skipping plot.")


if __name__ == "__main__":
    main()
