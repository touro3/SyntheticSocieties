from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.tree import DecisionTreeRegressor, _tree
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required for persona_fidelity benchmark. "
        "Install it in the project venv before running this module."
    ) from exc


@dataclass
class ProfileTreeResult:
    frame: pd.DataFrame
    profiles_df: pd.DataFrame
    profile_definitions: list[dict[str, Any]]
    feature_names: list[str]
    target_item_names: list[str]


def compute_composite_score(frame: pd.DataFrame, target_items: list[dict[str, Any]]) -> pd.Series:
    cols = []
    for item in target_items:
        series = frame[item["name"]].astype(float).clip(0.0, 1.0)
        if item.get("inverse", False):
            series = 1.0 - series
        cols.append(series)
    return pd.concat(cols, axis=1).mean(axis=1) * 100.0


def prepare_benchmark_frame(
    frame: pd.DataFrame,
    profile_features: list[dict[str, Any]],
    target_items: list[dict[str, Any]],
) -> pd.DataFrame:
    feature_names = [f["name"] for f in profile_features]
    target_names = [t["name"] for t in target_items]
    working = frame[feature_names + target_names].copy()
    working = working.dropna()
    working["benchmark_score_0_100"] = compute_composite_score(working, target_items)
    return working.reset_index(drop=True)


def fit_profile_tree(
    frame: pd.DataFrame,
    profile_features: list[dict[str, Any]],
    target_items: list[dict[str, Any]],
    min_support: int = 30,
    max_depth: int = 3,
    random_state: int = 42,
) -> ProfileTreeResult:
    feature_names = [f["name"] for f in profile_features]
    target_names = [t["name"] for t in target_items]

    X = frame[feature_names].to_numpy(dtype=float)
    y = frame["benchmark_score_0_100"].to_numpy(dtype=float)

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_support,
        random_state=random_state,
    )
    tree.fit(X, y)

    leaf_ids = tree.apply(X)
    working = frame.copy()
    working["leaf_id"] = leaf_ids

    rules_by_leaf = _extract_leaf_rules(tree.tree_, feature_names)
    rows = []
    definitions = []

    grouped = working.groupby("leaf_id", sort=True)
    for idx, (leaf_id, leaf_df) in enumerate(grouped, start=1):
        profile_id = f"P{idx:02d}"
        feature_summary = {}
        for feature in profile_features:
            name = feature["name"]
            feature_summary[name] = {
                "min": float(leaf_df[name].min()),
                "median": float(leaf_df[name].median()),
                "max": float(leaf_df[name].max()),
            }

        target_summary = {}
        for item in target_items:
            name = item["name"]
            target_summary[name] = {
                "mean": float(leaf_df[name].mean()),
                "std": float(leaf_df[name].std(ddof=1) if len(leaf_df) > 1 else 0.0),
            }

        score_mean = float(leaf_df["benchmark_score_0_100"].mean())
        score_std = float(leaf_df["benchmark_score_0_100"].std(ddof=1) if len(leaf_df) > 1 else 0.0)
        rule_bounds = rules_by_leaf[int(leaf_id)]

        profile_def = {
            "profile_id": profile_id,
            "leaf_id": int(leaf_id),
            "n_real": int(len(leaf_df)),
            "score_mean_real": score_mean,
            "score_std_real": score_std,
            "rules": _rule_strings(rule_bounds, feature_names),
            "rule_bounds": rule_bounds,
            "feature_summary": feature_summary,
            "target_summary": target_summary,
        }
        definitions.append(profile_def)

        rows.append(
            {
                "profile_id": profile_id,
                "leaf_id": int(leaf_id),
                "n_real": int(len(leaf_df)),
                "real_score_0_100": score_mean,
                **{f"real_{item['name']}": target_summary[item["name"]]["mean"] for item in target_items},
            }
        )

        working.loc[working["leaf_id"] == leaf_id, "profile_id"] = profile_id

    profiles_df = pd.DataFrame(rows).sort_values("profile_id").reset_index(drop=True)

    return ProfileTreeResult(
        frame=working,
        profiles_df=profiles_df,
        profile_definitions=definitions,
        feature_names=feature_names,
        target_item_names=target_names,
    )


def build_profile_text(profile_def: dict[str, Any], profile_features: list[dict[str, Any]]) -> str:
    feature_map = {f["name"]: f for f in profile_features}
    summary = profile_def["feature_summary"]

    lines = []
    for name, stats in summary.items():
        kind = feature_map[name].get("kind", "band_0_1")
        lines.append(_describe_feature(name, kind, stats))

    lines.append(
        f"This profile represents a subgroup with {profile_def['n_real']} real respondents in the source survey."
    )
    return "\n".join(f"- {line}" for line in lines if line)


def _describe_feature(name: str, kind: str, stats: dict[str, float]) -> str:
    lo = stats["min"]
    med = stats["median"]
    hi = stats["max"]

    if kind == "age":
        return f"You are approximately between {int(math.floor(lo))} and {int(math.ceil(hi))} years old."
    if kind == "gender":
        label = "male" if med <= 1.5 else "female"
        return f"You are {label}."
    if kind == "education":
        mapping = {
            1: "very low education",
            2: "lower secondary education",
            3: "upper secondary education",
            4: "post-secondary education",
            5: "short tertiary education",
            6: "bachelor-level education",
            7: "graduate-level education",
        }
        return f"Your education level is around {mapping.get(int(round(med)), 'upper secondary education')}."
    if kind == "income_decile":
        return f"Your household income is around decile {int(round(med))} out of 10."
    if kind == "urbanization":
        mapping = {
            1: "a big city",
            2: "the suburbs of a city",
            3: "a town",
            4: "a village",
            5: "the countryside",
        }
        return f"You live in or near {mapping.get(int(round(med)), 'a town')}."
    if kind == "religious_belonging":
        return "You report a religious affiliation." if med <= 1.5 else "You do not report a religious affiliation."
    if kind == "band_0_1":
        return f"Your {name.replace('_', ' ')} is {_band_word(med)}."
    return f"Your {name.replace('_', ' ')} is around {med:.2f}."


def _band_word(value: float) -> str:
    if value < 0.20:
        return "very low"
    if value < 0.40:
        return "low"
    if value < 0.60:
        return "moderate"
    if value < 0.80:
        return "high"
    return "very high"


def _extract_leaf_rules(tree_struct, feature_names: list[str]) -> dict[int, dict[str, dict[str, float]]]:
    rules: dict[int, dict[str, dict[str, float]]] = {}

    def recurse(node_id: int, bounds: dict[str, dict[str, float]]) -> None:
        feature_index = tree_struct.feature[node_id]
        if feature_index == _tree.TREE_UNDEFINED:
            rules[node_id] = {
                key: {"low": value["low"], "high": value["high"]}
                for key, value in bounds.items()
            }
            return

        feature_name = feature_names[feature_index]
        threshold = float(tree_struct.threshold[node_id])

        left_bounds = {
            key: {"low": value["low"], "high": value["high"]}
            for key, value in bounds.items()
        }
        right_bounds = {
            key: {"low": value["low"], "high": value["high"]}
            for key, value in bounds.items()
        }

        left_bounds.setdefault(feature_name, {"low": -float("inf"), "high": float("inf")})
        right_bounds.setdefault(feature_name, {"low": -float("inf"), "high": float("inf")})

        left_bounds[feature_name]["high"] = min(left_bounds[feature_name]["high"], threshold)
        right_bounds[feature_name]["low"] = max(right_bounds[feature_name]["low"], threshold)

        recurse(tree_struct.children_left[node_id], left_bounds)
        recurse(tree_struct.children_right[node_id], right_bounds)

    recurse(0, {})
    return rules


def _rule_strings(rule_bounds: dict[str, dict[str, float]], feature_names: list[str]) -> list[str]:
    rules = []
    for name in feature_names:
        if name not in rule_bounds:
            continue
        low = rule_bounds[name]["low"]
        high = rule_bounds[name]["high"]

        if np.isfinite(low) and np.isfinite(high):
            rules.append(f"{name}: ({low:.3f}, {high:.3f}]")
        elif np.isfinite(low):
            rules.append(f"{name}: > {low:.3f}")
        elif np.isfinite(high):
            rules.append(f"{name}: <= {high:.3f}")
    return rules
