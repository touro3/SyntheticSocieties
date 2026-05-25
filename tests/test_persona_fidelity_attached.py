"""Defect L-3 regression test.

`scripts/run_config_simulation.py:_attach_metrics_block` must emit
`summary["metrics"]["persona_fidelity"]` so that H8 / §8.5 memory ablation
can be measured. Prior to 2026-05-25 the function computed BRM and Gini
only; the on-disk memory-ablation runs therefore could not test the H8
prediction (Table 7). This test guards the wiring."""

from __future__ import annotations

from agents.profile import AgentProfile
from scripts.run_config_simulation import _attach_metrics_block


class _FakeAgent:
    """Minimal stand-in for `agents.agent.Agent` — _attach_metrics_block
    only accesses `agent.profile`."""

    def __init__(self, profile: AgentProfile) -> None:
        self.profile = profile


def _make_agent(aid: str, trust: float, risk: float, social: float) -> _FakeAgent:
    return _FakeAgent(
        AgentProfile(
            agent_id=aid,
            age=35,
            income=30_000.0,
            education="bachelor",
            occupation="researcher",
            location="AT",
            political_preference="centrist",
            risk_tolerance=risk,
            social_class="middle",
            trust_people=trust,
            social_activity=social,
        )
    )


def _make_events(agent_ids: list[str], rounds: int) -> list[dict]:
    """Round-by-round cooperate/work/save events with deterministic mix."""
    out: list[dict] = []
    for r in range(1, rounds + 1):
        for i, aid in enumerate(agent_ids):
            # Rotate actions so the per-agent cooperate rate is non-degenerate.
            action_type = ("cooperate", "work", "save")[(r + i) % 3]
            out.append(
                {
                    "round_id": r,
                    "agent_id": aid,
                    "action": {"action_type": action_type, "target_agent_id": None, "amount": 1.0},
                    "state_after": {"wealth": 100.0 + r, "stress": 0.1, "satisfaction": 0.5},
                }
            )
    return out


def _make_summary(wealth_per_agent: list[float], action_totals: dict[str, int]) -> dict:
    return {
        "wealth": {"values": list(wealth_per_agent)},
        "event_action_counts": dict(action_totals),
    }


def test_persona_fidelity_lands_in_summary_when_agents_passed():
    agents = [
        _make_agent("agent_0", trust=0.7, risk=0.3, social=0.6),
        _make_agent("agent_1", trust=0.4, risk=0.5, social=0.4),
        _make_agent("agent_2", trust=0.2, risk=0.8, social=0.2),
    ]
    rounds = 10
    events = _make_events([a.profile.agent_id for a in agents], rounds=rounds)
    # Total event counts: each agent makes 10 actions evenly across three types.
    total_per_action = (rounds * len(agents)) // 3 + 1
    summary = _make_summary(
        wealth_per_agent=[100.0 + i * 10 for i in range(len(agents))],
        action_totals={"work": total_per_action, "save": total_per_action, "cooperate": total_per_action},
    )

    _attach_metrics_block(summary, events, {"llm": {"ablation_level": 5}}, agents=agents)

    assert "metrics" in summary, "metrics block must be attached"
    assert "persona_fidelity" in summary["metrics"], "L-3: persona_fidelity must be emitted"
    pf = summary["metrics"]["persona_fidelity"]
    assert pf is not None, "persona_fidelity must not be None when agents produce cooperate events"
    assert 0.0 <= pf <= 1.0, f"persona_fidelity must be in [0, 1], got {pf}"
    assert "persona_fidelity_decay_rate" in summary["metrics"]
    assert "persona_fidelity_per_agent" in summary
    assert set(summary["persona_fidelity_per_agent"].keys()) == {a.profile.agent_id for a in agents}


def test_persona_fidelity_back_compat_when_agents_omitted():
    """Calling _attach_metrics_block without agents (old signature) must not
    crash and must simply skip persona_fidelity — guards the pre-2026-05-25
    backfill path used by analysis scripts."""
    agents = [_make_agent("agent_0", 0.5, 0.5, 0.5)]
    events = _make_events(["agent_0"], rounds=5)
    summary = _make_summary([100.0], {"work": 2, "save": 2, "cooperate": 1})

    _attach_metrics_block(summary, events, {"llm": {"ablation_level": 0}})

    assert "metrics" in summary
    assert "brm" in summary["metrics"], "BRM must still be computed on old signature"
    assert "persona_fidelity" not in summary["metrics"], (
        "persona_fidelity must be silently skipped when agents not supplied "
        "(back-compat for legacy callers)"
    )
