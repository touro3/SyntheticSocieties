"""Tests for scripts/run_policy_sweep.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.policy_intervention import SweepPoint, sensitivity_index
from metrics.policy_sensitivity import ClusterOutcome, direction_recovery
from scripts.run_policy_sweep import (
    _coop_rate,
    dry_run_sweep,
    run_sweep,
    save_results,
)
from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Action collapse detected.*:UserWarning"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent(agent_id: str = "a0", wealth: float = 50.0, last_action=None) -> Agent:
    profile = AgentProfile(
        agent_id=agent_id,
        age=35,
        income=1000.0,
        education="secondary",
        occupation="worker",
        location="EU",
        political_preference="centrist",
        risk_tolerance=0.5,
        social_class="middle",
    )
    state = AgentState(wealth=wealth)
    state.last_action = last_action
    return Agent(profile=profile, state=state, memory=MemoryBuffer(max_items=10),
                 policy=MockPolicy())


# ── test_sweep_returns_six_points ─────────────────────────────────────────────

def test_sweep_returns_six_points():
    results = run_sweep(n_agents=6, n_rounds=4)
    assert len(results) == 6, f"Expected 6 SweepPoints, got {len(results)}"


# ── test_sweep_points_gini_in_unit_interval ───────────────────────────────────

def test_sweep_points_gini_in_unit_interval():
    results = run_sweep(n_agents=6, n_rounds=4)
    for sp in results:
        assert 0.0 <= sp.post_gini <= 1.0, (
            f"post_gini out of [0,1] for param={sp.parameter}: {sp.post_gini}"
        )


# ── test_sensitivity_index_negative_for_redistribute ──────────────────────────

def test_sensitivity_index_negative_for_redistribute():
    results = run_sweep(rule="redistribute_top_pct", n_agents=6, n_rounds=4)
    si = sensitivity_index(results, "gini")
    assert si <= 0, (
        f"Expected negative sensitivity index for redistribute_top_pct, got {si}"
    )


# ── test_direction_recovery_from_sweep_results ────────────────────────────────

def test_direction_recovery_from_sweep_results():
    # Use deterministic dry-run data to validate direction_recovery integration.
    # The dry_run_sweep produces monotonically decreasing Gini — the expected direction.
    results = dry_run_sweep(rule="redistribute_top_pct")
    policy_pairs = [
        (sp.parameter, ClusterOutcome(
            cluster_name="policy_sweep",
            simulated_gini=sp.post_gini,
            simulated_coop=sp.post_coop_rate,
        ))
        for sp in results
    ]
    dr = direction_recovery([], policy_parameter_pairs=policy_pairs)
    sweep_checks = [r for r in dr if "redistribution" in r.check]
    assert len(sweep_checks) >= 1, "Expected at least one redistribution direction check"
    assert sweep_checks[0].recovered, (
        f"Direction not recovered: {sweep_checks[0].check} (Δ={sweep_checks[0].delta})"
    )


# ── test_save_results_writes_valid_json ───────────────────────────────────────

def test_save_results_writes_valid_json(tmp_path):
    results = dry_run_sweep()
    out = tmp_path / "sweep.json"
    save_results(results, "redistribute_top_pct", out)
    assert out.exists()
    with open(out) as f:
        data = json.loads(f.read())
    assert "sweep_points" in data
    assert len(data["sweep_points"]) == 6
    assert "sensitivity_index_gini" in data


# ── test_coop_rate_helper ─────────────────────────────────────────────────────

def test_coop_rate_helper_all_cooperate():
    agents = [_make_agent(f"a{i}", last_action="cooperate") for i in range(5)]
    assert _coop_rate(agents) == 1.0


def test_coop_rate_helper_none_cooperate():
    agents = [_make_agent(f"a{i}", last_action="work") for i in range(4)]
    assert _coop_rate(agents) == 0.0


def test_coop_rate_helper_no_actions():
    agents = [_make_agent(f"a{i}", last_action=None) for i in range(3)]
    assert _coop_rate(agents) == 0.0
