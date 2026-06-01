"""Tests for the observational trajectory + pattern bank (ruflo ReasoningBank)."""

from __future__ import annotations

from types import SimpleNamespace

from metrics.trajectory_bank import TrajectoryBank


def _agent(agent_id="agent_0", wealth=50.0, social_class="middle"):
    return SimpleNamespace(
        state=SimpleNamespace(wealth=wealth),
        profile=SimpleNamespace(agent_id=agent_id, social_class=social_class),
    )


def test_record_writes_jsonl_rows(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    with TrajectoryBank(path) as bank:
        bank.record(_agent(), "work", {"wealth_delta": 8}, round_id=1)
        bank.record(_agent(), "steal", {"wealth_delta": -5}, round_id=2)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert '"verdict": "good"' in lines[0]
    assert '"verdict": "bad"' in lines[1]


def test_top_patterns_ranks_by_success_rate(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    bank = TrajectoryBank(path)
    # 'work' from mid wealth always good; 'steal' always bad.
    for _ in range(3):
        bank.record(_agent(wealth=50), "work", {"wealth_delta": 8}, 1)
    for _ in range(2):
        bank.record(_agent(wealth=50), "steal", {"wealth_delta": -3}, 1)
    bank.close()

    patterns = TrajectoryBank.top_patterns(path)
    assert patterns[0]["action"] == "work"
    assert patterns[0]["success_rate"] == 1.0
    assert patterns[0]["count"] == 3
    steal = next(p for p in patterns if p["action"] == "steal")
    assert steal["success_rate"] == 0.0


def test_top_patterns_missing_file_returns_empty(tmp_path):
    assert TrajectoryBank.top_patterns(tmp_path / "nope.jsonl") == []


def test_round_processor_inert_without_bank():
    """Default RoundProcessor must not require or use a trajectory bank."""
    from simulation.round_processor import RoundProcessor

    rp = RoundProcessor(world=None, agent_lookup={}, logger=None)
    assert rp.trajectory_bank is None
