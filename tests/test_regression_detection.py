"""Tests for rolling-window metric regression detection (ruflo perf check)."""

from __future__ import annotations

import pandas as pd

from tracker.analytics import detect_regression


def _write_index(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "experiment_index.parquet"
    df.to_parquet(path)
    return str(path)


def test_detects_metric_drop_after_stable_window(tmp_path):
    rows = [{"experiment_id": f"exp_{i}", "policy_type": "llm", "seed": i, "wealth_mean": 100.0} for i in range(6)]
    # Stable at 100, then a sharp regression.
    rows.append({"experiment_id": "exp_bad", "policy_type": "llm", "seed": 6, "wealth_mean": 40.0})
    idx = _write_index(tmp_path, rows)

    flagged = detect_regression(metric="wealth_mean", index_path=idx, window=5)
    assert any(f["experiment_id"] == "exp_bad" for f in flagged)
    bad = next(f for f in flagged if f["experiment_id"] == "exp_bad")
    assert bad["direction"] == "below"
    assert bad["policy_type"] == "llm"


def test_no_false_positive_on_stable_series(tmp_path):
    rows = [
        {"experiment_id": f"exp_{i}", "policy_type": "llm", "seed": i, "wealth_mean": 100.0 + (i % 2)}
        for i in range(10)
    ]
    idx = _write_index(tmp_path, rows)
    flagged = detect_regression(metric="wealth_mean", index_path=idx, window=5)
    assert flagged == []


def test_policies_are_isolated(tmp_path):
    rows = []
    for i in range(6):
        rows.append({"experiment_id": f"a_{i}", "policy_type": "llm", "seed": i, "wealth_mean": 100.0})
    for i in range(6):
        rows.append({"experiment_id": f"b_{i}", "policy_type": "random", "seed": i, "wealth_mean": 10.0})
    idx = _write_index(tmp_path, rows)
    # 'random' is consistently low but stable within its own policy → no flag.
    flagged = detect_regression(metric="wealth_mean", index_path=idx, window=5)
    assert flagged == []
