"""Integration tests for Phase 4 fixes.

Tests critical paths that span multiple modules:
- Per-round metrics logging (kernel.round_metrics)
- Trust updates from cooperation
- Dynamic network evolution
- Satisfaction dynamics
- Seed hierarchy determinism
- Token budget in experimental prompt builder
"""

from __future__ import annotations

import warnings

import pytest
from conftest import make_agent

from bgf_logging.event_logger import EventLogger
from configs.seed_manager import SeedManager
from decision.schemas import ProposedAction
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from simulation.kernel import SimulationKernel

# ── Helpers ──────────────────────────────────────────────────────────────────


class _WorkPolicy:
    """Deterministic policy that always works."""

    def propose_action(self, profile, state, memory, context, round_id):
        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="test",
            confidence=1.0,
        )


class _CooperatePolicy:
    """Deterministic policy that cooperates with first neighbor."""

    def __init__(self):
        from decision.graph_rag import GraphRAG

        self.graph_rag = GraphRAG()

    def propose_action(self, profile, state, memory, context, round_id):
        neighbors = context.get("network", {}).get("neighbors", [])
        if neighbors:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=5.0,
                reasoning_summary="test cooperation",
                confidence=1.0,
            )
        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="no neighbors",
            confidence=1.0,
        )


def _make_kernel(tmp_path, policy, n_agents=3):
    agents = [make_agent(f"agent_{i}", wealth=100.0) for i in range(n_agents)]
    for a in agents:
        a.policy = policy
    network = NetworkManager.fully_connected([a.profile.agent_id for a in agents])
    world = World(
        state=WorldState(round_id=0),
        institution_manager=InstitutionManager(),
        network_manager=network,
    )
    logger = EventLogger(tmp_path / "events.jsonl", overwrite=True)
    return SimulationKernel(agents=agents, world=world, logger=logger)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Per-round metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestKernelRoundMetrics:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_round_metrics_populated(self, tmp_path):
        """Kernel must populate round_metrics after run()."""
        kernel = _make_kernel(tmp_path, _WorkPolicy(), n_agents=3)
        kernel.run(num_rounds=3)
        assert len(kernel.round_metrics) == 3

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_round_metrics_has_expected_keys(self, tmp_path):
        kernel = _make_kernel(tmp_path, _WorkPolicy(), n_agents=3)
        kernel.run(num_rounds=1)
        m = kernel.round_metrics[0]
        for key in [
            "round_id",
            "action_distribution",
            "gini",
            "mean_wealth",
            "mean_stress",
            "mean_satisfaction",
            "n_agents",
        ]:
            assert key in m, f"Missing key: {key}"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_action_collapse_warning_emitted(self, tmp_path):
        """When all agents pick same action, a UserWarning is emitted."""
        kernel = _make_kernel(tmp_path, _WorkPolicy(), n_agents=5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kernel.run(num_rounds=1)
        collapse_warnings = [w for w in caught if "Action collapse" in str(w.message)]
        assert len(collapse_warnings) >= 1

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_gini_is_zero_for_equal_wealth(self, tmp_path):
        """All agents start at same wealth → Gini should be 0."""
        kernel = _make_kernel(tmp_path, _WorkPolicy(), n_agents=3)
        kernel.run(num_rounds=1)
        # After 1 round of equal work, all agents have same wealth
        assert kernel.round_metrics[0]["gini"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Trust updates from cooperation
# ═══════════════════════════════════════════════════════════════════════════════


class TestCooperationTrustUpdates:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_cooperation_populates_trust_dict(self, tmp_path):
        """After cooperation, both agents should have non-empty trust dicts."""
        kernel = _make_kernel(tmp_path, _CooperatePolicy(), n_agents=2)
        kernel.run(num_rounds=1)
        a0, a1 = kernel.agents
        # At least one agent should have trust toward the other
        assert a0.state.trust or a1.state.trust

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_trust_increases_with_repeated_cooperation(self, tmp_path):
        """Multiple rounds of cooperation should increase trust."""
        kernel = _make_kernel(tmp_path, _CooperatePolicy(), n_agents=2)
        kernel.run(num_rounds=1)
        trust_after_1 = dict(kernel.agents[0].state.trust)
        kernel.run(num_rounds=2)
        trust_after_3 = dict(kernel.agents[0].state.trust)
        # Trust should be higher after more cooperation
        for partner, t in trust_after_3.items():
            if partner in trust_after_1:
                assert t >= trust_after_1[partner]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Dynamic network evolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestDynamicNetworkEvolution:
    def test_network_add_edge(self):
        """add_edge creates new edges in the network."""
        net = NetworkManager.fully_connected(["a", "b", "c"])
        initial_edges = net.num_edges()
        # Adding existing edge should strengthen, not duplicate
        net.add_edge("a", "b")
        assert net.num_edges() == initial_edges
        assert net.get_edge_weight("a", "b") > 1.0

    def test_network_strengthen_edge(self):
        """strengthen_edge increases weight."""
        net = NetworkManager.fully_connected(["a", "b"])
        net.strengthen_edge("a", "b", increment=0.5)
        assert net.get_edge_weight("a", "b") == 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Satisfaction dynamics
# ═══════════════════════════════════════════════════════════════════════════════


class TestSatisfactionDynamics:
    def test_work_affects_satisfaction(self, tmp_path):
        kernel = _make_kernel(tmp_path, _WorkPolicy(), n_agents=1)
        initial_sat = kernel.agents[0].state.satisfaction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel.run(num_rounds=1)
        assert kernel.agents[0].state.satisfaction != initial_sat

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_cooperation_gives_highest_satisfaction(self):
        """Cooperation should give the highest satisfaction_delta."""
        manager = InstitutionManager()
        ws = WorldState(round_id=1)
        source = make_agent("a1", wealth=100.0)
        target = make_agent("a2", wealth=50.0)
        lookup = {"a1": source, "a2": target}

        # Work
        work_event = manager.execute(
            ProposedAction(action_type="work", amount=10.0, reasoning_summary="earn"),
            source,
            ws,
            lookup,
        )
        # Cooperate
        coop_event = manager.execute(
            ProposedAction(
                action_type="cooperate",
                target_agent_id="a2",
                amount=5.0,
                reasoning_summary="help",
            ),
            source,
            ws,
            lookup,
        )
        assert coop_event["satisfaction_delta"] > work_event["satisfaction_delta"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Seed hierarchy
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeedHierarchy:
    def test_same_master_same_seeds(self):
        """Same master seed → identical component seeds."""
        s1 = SeedManager(42)
        s2 = SeedManager(42)
        assert s1.all_seeds() == s2.all_seeds()

    def test_different_master_different_seeds(self):
        """Different master seeds → different component seeds."""
        s1 = SeedManager(42)
        s2 = SeedManager(123)
        assert s1.population_seed() != s2.population_seed()
        assert s1.network_seed() != s2.network_seed()

    def test_component_seeds_are_independent(self):
        """Different components must produce different seeds."""
        s = SeedManager(42)
        seeds = s.all_seeds()
        seed_values = [v for k, v in seeds.items() if k != "master"]
        assert len(set(seed_values)) == len(seed_values), "Component seeds must be unique"

    def test_all_seeds_returns_dict(self):
        s = SeedManager(42)
        d = s.all_seeds()
        assert isinstance(d, dict)
        assert "master" in d
        assert "population" in d
        assert "network" in d
        assert "llm" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Population helpers deduplication
# ═══════════════════════════════════════════════════════════════════════════════


class TestPopulationHelpers:
    def test_shared_helpers_importable(self):
        """_helpers module must be importable."""
        from population._helpers import (
            clamp01,
            map_education,
            map_location,
            map_political,
            map_social_class,
            safe_float,
            safe_int,
        )

        assert safe_float("3.14") == 3.14
        assert safe_int("5") == 5
        assert clamp01(1.5) == 1.0
        assert map_education(6) == "bachelor"
        assert map_location(1) == "big_city"
        assert map_political(0.1) == "left"
        assert map_social_class(2) == "lower"

    def test_generator_uses_shared_helpers(self):
        """generator.py must import from _helpers."""
        import population.generator as gen

        # The module should have the aliased functions
        assert hasattr(gen, "_safe_float")
        assert hasattr(gen, "_map_education")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: AblationLevel IntEnum
# ═══════════════════════════════════════════════════════════════════════════════


class TestAblationLevelIntEnum:
    def test_is_intenum(self):
        from enum import IntEnum

        from decision.prompt_builder import AblationLevel

        assert issubclass(AblationLevel, IntEnum)

    def test_comparison_still_works(self):
        from decision.prompt_builder import AblationLevel

        assert AblationLevel.FULL >= AblationLevel.BASELINE
        assert AblationLevel.BALANCED >= 4
        assert AblationLevel.BASELINE == 0

    def test_iteration(self):
        from decision.prompt_builder import AblationLevel

        levels = list(AblationLevel)
        assert len(levels) == 6
        assert levels[0].value == 0
        assert levels[-1].value == 5
