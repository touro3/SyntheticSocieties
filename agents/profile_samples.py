import numpy as np
import polars as pl


class ProfileSampler:

    def __init__(self, data_dir="data/socioeconomic"):
        self.income_dist = pl.read_csv(f"{data_dir}/income_distribution.csv")
        self.education_dist = pl.read_csv(f"{data_dir}/education_distribution.csv")
        self.occupation_dist = pl.read_csv(f"{data_dir}/occupation_distribution.csv")

    def sample_income(self):
        rows = self.income_dist.to_dicts()
        probs = [r["probability"] for r in rows]
        bracket = np.random.choice(rows, p=probs)

        low, high = bracket["income_brackets"].split("-")

        low = float(low)
        high = float(high.replace("+","5000"))

        return np.random.uniform(low, high)

    def sample_education(self):
        rows = self.education_dist.to_dicts()
        probs = [r["probability"] for r in rows]

        return np.random.choice(
            [r["education"] for r in rows],
            p=probs
        )

    def sample_occupation(self):
        rows = self.occupation_dist.to_dicts()
        probs = [r["probability"] for r in rows]

        occ = np.random.choice(rows, p=probs)

        return occ["occupation"], occ["income_multiplier"]

    def sample_risk(self):
        return np.clip(np.random.normal(0.5, 0.15), 0, 1)