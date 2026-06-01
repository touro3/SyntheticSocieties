"""Tests for scoped analytics filtering in tracker/analytics.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tracker.analytics import compare_llm_vs_baselines, query_by_policy


def _make_index(path: Path) -> None:
    df = pd.DataFrame(
        [
            # Scoped cmp run set (seeds 1,2)
            {
                "experiment_id": "cmp_llm_s1",
                "policy_type": "llm",
                "seed": 1,
                "wealth_mean": 200.0,
                "wealth_gini": 0.20,
                "stress_mean": 0.8,
            },
            {
                "experiment_id": "cmp_llm_s2",
                "policy_type": "llm",
                "seed": 2,
                "wealth_mean": 220.0,
                "wealth_gini": 0.18,
                "stress_mean": 0.9,
            },
            {
                "experiment_id": "cmp_template_s1",
                "policy_type": "template",
                "seed": 1,
                "wealth_mean": 150.0,
                "wealth_gini": 0.30,
                "stress_mean": 0.7,
            },
            {
                "experiment_id": "cmp_template_s2",
                "policy_type": "template",
                "seed": 2,
                "wealth_mean": 155.0,
                "wealth_gini": 0.31,
                "stress_mean": 0.75,
            },
            {
                "experiment_id": "cmp_rule_s1",
                "policy_type": "rule_based",
                "seed": 1,
                "wealth_mean": 110.0,
                "wealth_gini": 0.17,
                "stress_mean": 1.0,
            },
            {
                "experiment_id": "cmp_rule_s2",
                "policy_type": "rule_based",
                "seed": 2,
                "wealth_mean": 108.0,
                "wealth_gini": 0.18,
                "stress_mean": 1.1,
            },
            {
                "experiment_id": "cmp_random_s1",
                "policy_type": "random",
                "seed": 1,
                "wealth_mean": 140.0,
                "wealth_gini": 0.11,
                "stress_mean": 0.6,
            },
            {
                "experiment_id": "cmp_random_s2",
                "policy_type": "random",
                "seed": 2,
                "wealth_mean": 142.0,
                "wealth_gini": 0.12,
                "stress_mean": 0.62,
            },
            # Historical / out-of-scope rows that must be excluded by filter
            {
                "experiment_id": "cmp_llm_s42",
                "policy_type": "llm",
                "seed": 42,
                "wealth_mean": 999.0,
                "wealth_gini": 0.01,
                "stress_mean": 9.9,
            },
            {
                "experiment_id": "ablation_no_persona_s1",
                "policy_type": "ablated_llm",
                "seed": 1,
                "wealth_mean": 120.0,
                "wealth_gini": 0.08,
                "stress_mean": 1.2,
            },
        ]
    )
    df.to_parquet(path, index=False)


def test_query_by_policy_scoped_filters(tmp_path: Path):
    index_path = tmp_path / "index.parquet"
    _make_index(index_path)

    out = query_by_policy(
        index_path=str(index_path),
        experiment_ids=[
            "cmp_llm_s1",
            "cmp_llm_s2",
            "cmp_template_s1",
            "cmp_template_s2",
            "cmp_rule_s1",
            "cmp_rule_s2",
            "cmp_random_s1",
            "cmp_random_s2",
        ],
        seeds=[1, 2],
        policy_types=["llm", "template", "rule_based", "random"],
        require_cmp_only=True,
    )

    by_policy = {r["policy_type"]: r for _, r in out.iterrows()}
    assert set(by_policy) == {"llm", "template", "rule_based", "random"}
    assert by_policy["llm"]["n_experiments"] == 2
    assert abs(by_policy["llm"]["avg_wealth_mean"] - 210.0) < 1e-9


def test_compare_llm_vs_baselines_scoped_counts(tmp_path: Path):
    index_path = tmp_path / "index.parquet"
    _make_index(index_path)

    out = compare_llm_vs_baselines(
        index_path=str(index_path),
        seeds=[1, 2],
        policy_types=["llm", "template", "rule_based", "random"],
        require_cmp_only=True,
    )
    by_policy = {r["policy_type"]: r for _, r in out.iterrows()}
    assert by_policy["llm"]["n_runs"] == 2
    assert by_policy["template"]["n_runs"] == 2
    assert by_policy["rule_based"]["n_runs"] == 2
    assert by_policy["random"]["n_runs"] == 2
