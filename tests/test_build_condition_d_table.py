"""Tests for the Condition D (Rule-Based ESS) Table-3b builder (28.2).

Builds the no-LLM calibration-anchor row from on-disk experiment dirs and
aggregates per-seed coop_rate / Gini / B_RLHF with the shared BCa 95% CI.
A synthetic experiment-dir fixture keeps this GPU- and sim-free.
"""

from __future__ import annotations

import json

import pytest

from metrics.behavioral_realism import rlhf_bias_index_from_counts
from scripts.build_condition_d_table import build, load_condition_d_run


def _make_run(root, name, *, n=100, t=30, gini=0.5, work=100, save=50, cooperate=150):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "wealth": {"values": [1.0, 2.0, 3.0], "gini": gini},
                "event_action_counts": {"work": work, "save": save, "cooperate": cooperate},
            }
        )
    )
    (d / "metadata.json").write_text(
        json.dumps({"policy_type": "rule_based_ess", "population_size": n, "rounds": t, "seed": 1})
    )
    return d


class TestLoadRun:
    def test_metrics_match_hand_computation(self, tmp_path):
        d = _make_run(tmp_path, "cd_s1", work=100, save=50, cooperate=150, gini=0.42)
        m = load_condition_d_run(d)
        assert m["gini"] == pytest.approx(0.42)
        assert m["coop_rate"] == pytest.approx(150 / 300)
        assert m["b_rlhf"] == pytest.approx(rlhf_bias_index_from_counts({"work": 100, "save": 50, "cooperate": 150}))
        assert m["population_size"] == 100 and m["rounds"] == 30

    def test_missing_dir_returns_none(self, tmp_path):
        assert load_condition_d_run(tmp_path / "nope") is None


class TestBuild:
    def test_aggregate_mean_and_bca(self, tmp_path):
        for i, g in enumerate((0.30, 0.32, 0.34), 1):
            _make_run(tmp_path, f"cd_s{i}", gini=g)
        res, md = build([1, 2, 3], prefix="cd_s", experiments_dir=tmp_path, out_json=tmp_path / "out.json")
        gini = res["per_metric"]["gini"]
        assert gini["n"] == 3
        assert gini["mean"] == pytest.approx(0.32, abs=1e-9)
        lo, hi = gini["ci95"]
        assert lo <= gini["mean"] <= hi

    def test_missing_seed_skipped_gracefully(self, tmp_path):
        _make_run(tmp_path, "cd_s1", gini=0.31)
        _make_run(tmp_path, "cd_s3", gini=0.33)
        res, _ = build([1, 2, 3], prefix="cd_s", experiments_dir=tmp_path, out_json=tmp_path / "o.json")
        assert res["per_metric"]["gini"]["n"] == 2
        assert res["experiment"]["seeds_found"] == [1, 3]

    def test_mixed_scale_raises(self, tmp_path):
        _make_run(tmp_path, "cd_s1", n=100)
        _make_run(tmp_path, "cd_s2", n=500)
        with pytest.raises(ValueError, match="[Mm]ixed|consistent|scale"):
            build([1, 2], prefix="cd_s", experiments_dir=tmp_path, out_json=tmp_path / "o.json")

    def test_emits_json_and_markdown_row(self, tmp_path):
        for i in (1, 2, 3):
            _make_run(tmp_path, f"cd_s{i}", gini=0.32)
        out = tmp_path / "condition_d_results.json"
        res, md = build([1, 2, 3], prefix="cd_s", experiments_dir=tmp_path, out_json=out)
        assert out.exists()
        disk = json.loads(out.read_text())
        assert disk["experiment"]["condition"] == "D"
        assert disk["status"] == "complete"
        for k in ("coop_rate", "gini", "b_rlhf"):
            assert "ci95" in disk["per_metric"][k] and "per_seed" in disk["per_metric"][k]
        cells = [c.strip() for c in md.strip().strip("|").split("|")]
        assert len(cells) == 6
        assert cells[1] == "D"
        assert cells[5] == "—"  # no within-D A/B baseline

    def test_zero_dirs_awaiting_runs(self, tmp_path):
        res, md = build([1, 2, 3], prefix="none_s", experiments_dir=tmp_path, out_json=tmp_path / "o.json")
        assert res["status"] == "awaiting_runs"
        assert md == ""
