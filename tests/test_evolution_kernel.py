"""Phase 1 — generational cultural-evolution loop over the Donor Game."""

from __future__ import annotations

import json

import pytest

from agents.agent import Agent
from agents.memory import HierarchicalMemory
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.mock_policy import MockPolicy
from environment.donor_game import DONATE, KEEP, DonorDecision
from simulation.evolution_kernel import EvolutionRunner, default_offspring_fn, default_strategy_fn


def _agent(aid: str, kind: str = "d") -> Agent:
    a = Agent(
        profile=AgentProfile(
            agent_id=aid,
            age=35,
            income=1000.0,
            education="secondary",
            occupation="worker",
            location="urban",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
        ),
        state=AgentState(wealth=0.0),
        memory=HierarchicalMemory(max_recent=10),
        policy=MockPolicy(),
    )
    a.kind = kind
    return a


def _by_kind(donor, recipient, rep, round_id) -> DonorDecision:
    return DonorDecision(DONATE if getattr(donor, "kind", "d") == "c" else KEEP)


def _kind_offspring(parent, new_id, inherited_strategy):
    child = default_offspring_fn(parent, new_id, inherited_strategy)
    child.kind = parent.kind  # heritable trait
    return child


# ── Selection ─────────────────────────────────────────────────────────────────


def test_selection_keeps_top_fraction():
    runner = EvolutionRunner(
        [_agent(f"a{i}") for i in range(4)],
        decision_fn=_by_kind,
        n_generations=3,
        rounds_per_generation=4,
        survivor_fraction=0.5,
        seed=1,
    )
    result = runner.run()
    for g in result.generations:
        assert len(g.survivor_ids) == 2  # top 50% of 4


def test_unconditional_cooperator_is_culled():
    # a0 always donates → strictly poorest → must be culled in gen 0.
    agents = [_agent("a0", "c"), _agent("a1", "d"), _agent("a2", "d"), _agent("a3", "d")]
    runner = EvolutionRunner(
        agents,
        decision_fn=_by_kind,
        n_generations=1,
        rounds_per_generation=12,
        offspring_fn=_kind_offspring,
        seed=2,
    )
    result = runner.run()
    assert "a0" in result.generations[0].culled_ids


def test_defectors_dominate_over_generations():
    agents = [_agent(f"c{i}", "c") for i in range(4)] + [_agent(f"d{i}", "d") for i in range(4)]
    runner = EvolutionRunner(
        agents,
        decision_fn=_by_kind,
        n_generations=6,
        rounds_per_generation=12,
        offspring_fn=_kind_offspring,
        seed=3,
    )
    result = runner.run()
    traj = result.cooperation_trajectory
    assert traj[0] > traj[-1]  # cooperation collapses under selection
    assert traj[-1] < 0.25


# ── Inheritance ───────────────────────────────────────────────────────────────


def test_offspring_inherit_strategy_text():
    runner = EvolutionRunner(
        [_agent(f"a{i}") for i in range(4)],
        decision_fn=_by_kind,
        n_generations=2,
        rounds_per_generation=4,
        strategy_fn=lambda agent, enc: "PARENT_STRAT",
        seed=4,
    )
    runner.run()
    offspring = [a for a in runner.agents if a.profile.agent_id.startswith("g")]
    assert offspring, "expected offspring after reproduction"
    for child in offspring:
        assert child.inherited_strategy == "PARENT_STRAT"
        assert child.parent_id is not None


def test_offspring_inherit_heritable_trait():
    # All defectors survive; offspring must also be defectors.
    agents = [_agent(f"c{i}", "c") for i in range(2)] + [_agent(f"d{i}", "d") for i in range(2)]
    runner = EvolutionRunner(
        agents,
        decision_fn=_by_kind,
        n_generations=4,
        rounds_per_generation=12,
        offspring_fn=_kind_offspring,
        seed=5,
    )
    runner.run()
    kinds = {a.kind for a in runner.agents}
    assert kinds == {"d"}  # cooperator lineage extinct


# ── Lineage + population invariants ───────────────────────────────────────────


def test_lineage_recorded_for_every_agent_every_generation():
    n_agents, n_gen = 6, 5
    runner = EvolutionRunner(
        [_agent(f"a{i}") for i in range(n_agents)],
        decision_fn=_by_kind,
        n_generations=n_gen,
        rounds_per_generation=4,
        seed=6,
    )
    result = runner.run()
    assert len(result.lineage) == n_agents * n_gen
    for gen in range(n_gen):
        gen_entries = [e for e in result.lineage if e.generation == gen]
        assert len(gen_entries) == n_agents
        assert gen_entries[0].fitness_rank == 0  # fittest recorded first


def test_population_size_constant():
    runner = EvolutionRunner(
        [_agent(f"a{i}") for i in range(8)],
        decision_fn=_by_kind,
        n_generations=4,
        rounds_per_generation=4,
        seed=7,
    )
    runner.run()
    assert len(runner.agents) == 8


def test_cooperation_trajectory_length_matches_generations():
    runner = EvolutionRunner(
        [_agent(f"a{i}") for i in range(4)],
        decision_fn=_by_kind,
        n_generations=7,
        rounds_per_generation=3,
        seed=8,
    )
    result = runner.run()
    assert len(result.cooperation_trajectory) == 7


# ── Strategy distillation ─────────────────────────────────────────────────────


def test_default_strategy_fn_summarizes_donor_behaviour():
    from environment.donor_game import EncounterRecord

    agent = _agent("a0", "c")
    enc = [
        EncounterRecord(0, "a0", "a1", DONATE, 2.0, 4.0, 0.9),
        EncounterRecord(0, "a0", "a2", DONATE, 2.0, 4.0, 0.8),
    ]
    text = default_strategy_fn(agent, enc)
    assert "100%" in text


# ── Validation + persistence ──────────────────────────────────────────────────


def test_invalid_survivor_fraction_rejected():
    with pytest.raises(ValueError):
        EvolutionRunner([_agent("a0"), _agent("a1")], survivor_fraction=1.5)


def test_save_results_writes_json(tmp_path):
    runner = EvolutionRunner(
        [_agent(f"a{i}") for i in range(4)],
        decision_fn=_by_kind,
        n_generations=2,
        rounds_per_generation=3,
        seed=9,
    )
    runner.run()
    out = tmp_path / "evo.json"
    runner.save_results(out)
    data = json.loads(out.read_text())
    assert "cooperation_trajectory" in data
    assert "generations" in data and "lineage" in data
    assert len(data["cooperation_trajectory"]) == 2


def test_reproducibility_same_seed():
    def _run():
        r = EvolutionRunner(
            [_agent(f"a{i}", "c" if i % 2 else "d") for i in range(6)],
            decision_fn=_by_kind,
            n_generations=4,
            rounds_per_generation=6,
            offspring_fn=_kind_offspring,
            seed=123,
        )
        return r.run().cooperation_trajectory

    assert _run() == _run()
