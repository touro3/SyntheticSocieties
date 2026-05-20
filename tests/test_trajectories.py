"""Tests for metrics/trajectories.py — trajectory extraction and multi-seed aggregation."""

import json

import numpy as np

from metrics.trajectories import aggregate_seeds, extract_trajectories

# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_events(path, events):
    """Write a list of event dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _make_event(round_id, agent_id, action_type, wealth, stress, target=None):
    return {
        "round_id": round_id,
        "agent_id": agent_id,
        "action": {"action_type": action_type, "target_agent_id": target},
        "state_after": {"wealth": wealth, "stress": stress, "last_action": action_type},
    }


# ── extract_trajectories ────────────────────────────────────────────────────


class TestExtractTrajectories:
    def test_empty_dir(self, tmp_path):
        result = extract_trajectories(tmp_path / "nonexistent")
        assert result == {}

    def test_basic_extraction(self, tmp_path):
        events = [
            _make_event(1, "a0", "work", 110.0, 1.0),
            _make_event(1, "a1", "save", 104.0, 0.0),
            _make_event(2, "a0", "cooperate", 107.0, 0.8, target="a1"),
            _make_event(2, "a1", "work", 114.0, 1.0),
        ]
        _write_events(tmp_path / "events.jsonl", events)
        result = extract_trajectories(tmp_path)

        assert result["rounds"] == [1, 2]
        assert len(result["wealth"][1]) == 2
        assert len(result["wealth"][2]) == 2
        assert result["stress"][1] == [1.0, 0.0]
        assert result["actions"][1]["work"] == 1
        assert result["actions"][1]["save"] == 1
        assert result["actions"][2]["cooperate"] == 1
        assert result["actions"][2]["work"] == 1

    def test_agent_trajectories(self, tmp_path):
        events = [
            _make_event(1, "a0", "work", 110.0, 1.0),
            _make_event(2, "a0", "save", 114.0, 0.8),
            _make_event(3, "a0", "cooperate", 111.0, 0.7, target="a1"),
        ]
        _write_events(tmp_path / "events.jsonl", events)
        result = extract_trajectories(tmp_path)

        agent_traj = result["agent_trajectories"]["a0"]
        assert agent_traj["wealth"] == [110.0, 114.0, 111.0]
        assert agent_traj["stress"] == [1.0, 0.8, 0.7]

    def test_malformed_json_skipped(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(_make_event(1, "a0", "work", 100.0, 0.5)) + "\n")
        result = extract_trajectories(tmp_path)
        assert result["rounds"] == [1]
        assert len(result["wealth"][1]) == 1

    def test_multiple_agents_per_round(self, tmp_path):
        events = [_make_event(1, f"a{i}", "work", 100.0 + i, float(i) * 0.1) for i in range(10)]
        _write_events(tmp_path / "events.jsonl", events)
        result = extract_trajectories(tmp_path)
        assert len(result["wealth"][1]) == 10
        assert result["actions"][1]["work"] == 10

    def test_returns_sorted_rounds(self, tmp_path):
        events = [
            _make_event(3, "a0", "work", 130.0, 3.0),
            _make_event(1, "a0", "work", 110.0, 1.0),
            _make_event(2, "a0", "save", 114.0, 0.8),
        ]
        _write_events(tmp_path / "events.jsonl", events)
        result = extract_trajectories(tmp_path)
        assert result["rounds"] == [1, 2, 3]


# ── aggregate_seeds ──────────────────────────────────────────────────────────


class TestAggregateSeeds:
    def _make_experiment(self, root, exp_id, n_agents=3, n_rounds=5, seed=None):
        """Create a synthetic experiment directory with events."""
        rng = np.random.RandomState(seed)
        exp_dir = root / exp_id
        events = []
        for r in range(1, n_rounds + 1):
            for i in range(n_agents):
                wealth = 100.0 + r * 10 + rng.normal(0, 5)
                stress = min(r * 0.2 + rng.normal(0, 0.1), 10.0)
                actions = ["work", "save", "cooperate"]
                action = actions[rng.randint(0, 3)]
                events.append(_make_event(r, f"a{i}", action, wealth, stress))
        _write_events(exp_dir / "events.jsonl", events)

    def test_single_seed(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_llm_s42", n_agents=3, n_rounds=5, seed=42)
        result = aggregate_seeds("llm", [42], experiments_root=tmp_path)
        assert result != {}
        assert result["n_seeds"] == 1
        assert len(result["wealth_mean"]) == 5
        assert len(result["stress_mean"]) == 5
        assert result["action_freqs"].shape == (5, 4)

    def test_multiple_seeds(self, tmp_path):
        for seed in [42, 123, 7]:
            self._make_experiment(tmp_path, f"cmp_llm_s{seed}", seed=seed)
        result = aggregate_seeds("llm", [42, 123, 7], experiments_root=tmp_path)
        assert result["n_seeds"] == 3

    def test_missing_seeds_skipped(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_llm_s42", seed=42)
        result = aggregate_seeds("llm", [42, 999, 888], experiments_root=tmp_path)
        assert result["n_seeds"] == 1

    def test_all_missing_returns_empty(self, tmp_path):
        result = aggregate_seeds("llm", [999], experiments_root=tmp_path)
        assert result == {}

    def test_action_freqs_sum_to_one(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_llm_s42", seed=42)
        result = aggregate_seeds("llm", [42], experiments_root=tmp_path)
        row_sums = result["action_freqs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_different_round_lengths_trimmed(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_llm_s42", n_rounds=10, seed=42)
        self._make_experiment(tmp_path, "cmp_llm_s7", n_rounds=5, seed=7)
        result = aggregate_seeds("llm", [42, 7], experiments_root=tmp_path)
        # Trimmed to min rounds (5)
        assert len(result["wealth_mean"]) == 5
        assert result["n_seeds"] == 2

    def test_action_labels_present(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_llm_s42", seed=42)
        result = aggregate_seeds("llm", [42], experiments_root=tmp_path)
        assert result["action_labels"] == ["work", "save", "cooperate", "steal"]

    def test_template_policy_prefix(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_template_s42", seed=42)
        result = aggregate_seeds("template", [42], experiments_root=tmp_path)
        assert result["n_seeds"] == 1

    def test_rule_based_policy_prefix(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_rule_s42", seed=42)
        result = aggregate_seeds("rule_based", [42], experiments_root=tmp_path)
        assert result["n_seeds"] == 1

    def test_pool_wealth_returned(self, tmp_path):
        self._make_experiment(tmp_path, "cmp_llm_s42", n_agents=3, n_rounds=5, seed=42)
        result = aggregate_seeds("llm", [42], experiments_root=tmp_path)
        assert "pool_wealth" in result
        assert "pool_stress" in result
        assert result["pool_wealth"].shape[0] == 5  # rounds
        assert result["pool_wealth"].shape[1] == 3  # agents

    def test_pool_wealth_multi_seed_concatenated(self, tmp_path):
        for seed in [42, 7]:
            self._make_experiment(tmp_path, f"cmp_llm_s{seed}", n_agents=4, n_rounds=5, seed=seed)
        result = aggregate_seeds("llm", [42, 7], experiments_root=tmp_path)
        assert result["pool_wealth"].shape == (5, 8)  # 4 agents * 2 seeds
