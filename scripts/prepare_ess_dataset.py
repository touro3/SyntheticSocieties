from pathlib import Path

import pandas as pd

INPUT = "data/ESS11INTe04_1.csv"
OUTPUT = "data/behavior/ess_behavior_dataset.csv"


def load_dataset():

    df = pd.read_csv(INPUT)

    print("Raw shape:", df.shape)

    return df


def select_features(df):

    df = df[
        [
            "cntry",
            "intagea",
            "intgndr",
        ]
    ]

    df = df.rename(
        columns={
            "cntry": "country",
            "intagea": "age",
            "intgndr": "gender",
        }
    )

    return df


def clean_dataset(df):

    df = df.dropna()

    df = df[df["age"] > 18]

    return df


def generate_behavior_proxy(df):
    """
    Create a simple behavioral proxy.
    This is necessary because the interview dataset
    does not contain economic actions.
    """

    def behavior(row):

        if row["age"] < 30:
            return "work"

        if row["age"] > 60:
            return "save"

        if row["gender"] == 2:
            return "cooperate"

        return "work"

    df["action"] = df.apply(behavior, axis=1)

    return df


def main():

    df = load_dataset()

    df = select_features(df)

    df = clean_dataset(df)

    df = generate_behavior_proxy(df)

    Path("data/behavior").mkdir(exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    print("Saved cleaned dataset:")
    print(OUTPUT)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()
