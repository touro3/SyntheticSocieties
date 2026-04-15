from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from agents.profile import AgentProfile


class EmpiricalProfileLoader:
    """
    Loads validated profiles from a fidelity benchmark artifact directory
    and generates a proportionally representative synthetic population.
    """

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

    def load_population(self, target_size: int, seed: int = 42) -> list[AgentProfile]:
        """Generates a population of AgentProfiles proportional to the real ESS data."""
        rng = random.Random(seed)

        # Load the profile definitions from the Phase A benchmark
        defs_path = self.artifact_dir / "profile_definitions.json"
        with open(defs_path) as f:
            profiles_data = json.load(f)

        # Load real distribution to get the proportion weights (n_real)
        summary_path = self.artifact_dir / "real_profile_summary.csv"
        df_summary = pd.read_csv(summary_path)

        # Calculate exact number of agents per profile based on true distribution
        total_real = df_summary["n_real"].sum()
        df_summary["proportion"] = df_summary["n_real"] / total_real
        df_summary["target_count"] = (df_summary["proportion"] * target_size).round().astype(int)

        # Adjust rounding errors to match target_size exactly
        while df_summary["target_count"].sum() < target_size:
            idx = rng.choice(df_summary.index)
            df_summary.loc[idx, "target_count"] += 1
        while df_summary["target_count"].sum() > target_size:
            idx = rng.choice(df_summary[df_summary["target_count"] > 0].index)
            df_summary.loc[idx, "target_count"] -= 1

        population = []
        agent_counter = 1

        for _, row in df_summary.iterrows():
            p_id = row["profile_id"]
            count = row["target_count"]

            # Find matching definition
            p_def = next((p for p in profiles_data if p["profile_id"] == p_id), None)
            if not p_def:
                continue

            for _ in range(count):
                # Map ESS features to AgentProfile v0.2
                # We add slight numerical jitter to income to prevent identical initial states
                base_income = float(p_def.get("income_decile", 5)) * 10.0 + rng.uniform(-2, 2)

                profile = AgentProfile(
                    agent_id=f"agent_{agent_counter:04d}",
                    age=p_def.get("age", 40),
                    income=base_income,
                    education=str(p_def.get("education_level", "secondary")),
                    occupation="worker",  # Default fallback
                    location="urban",  # Default fallback
                    political_preference="center",  # Overridden by ESS specific traits
                    risk_tolerance=0.5,
                    social_class=f"decile_{p_def.get('income_decile', 5)}",
                    # ESS-derived grounded attributes
                    gender=p_def.get("gender"),
                    country=p_def.get("country", "EU"),
                    income_decile=p_def.get("income_decile"),
                    trust_people=p_def.get("trust_people", 0.5),
                    trust_institutions=p_def.get("trust_institutions", 0.5),
                    political_orientation=p_def.get("left_right", 0.5),
                    life_satisfaction=p_def.get("life_satisfaction", 0.5),
                    happiness=p_def.get("happiness", 0.5),
                    immigration_attitude=p_def.get("immigration_same_ethnicity", 0.5),
                )
                population.append(profile)
                agent_counter += 1

        rng.shuffle(population)
        return population
