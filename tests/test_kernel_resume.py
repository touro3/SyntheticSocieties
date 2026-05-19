"""Tests for SimulationKernel heartbeat, checkpoint, and resume features.

No GPU required — all tests use MockPolicy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.agent import Agent
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from bgf_logging.event_logger import EventLogger
from decision.mock_policy import MockPolicy
from environment.institutions import InstitutionManager
from environment.world import World
from environment.world_state import WorldState
from simulation.kernel import SimulationKernel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(agent_id: str, wealth: float = 50.0) -> Agent:
    return Agent(
        profile=AgentProfile(
            agent_id=agent_id,
            age=30,
            income=1000,
            education="college",
            occupation="worker",
            location="italy",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
        ),
        state=AgentState(wealth=wealth),
        memory=MemoryBuffer(max_items=5),
        policy=MockPolicy(),
    )


def _make_world() -> World:
    return World(
        state=WorldState(
            public_signal={"economy": "stable"},
            prices={"food": 1.0},
            resources={"jobs": 100.0},
        ),
        institution_manager=InstitutionManager(),
    )


def _make_kernel(
    agents=None,
    *,
    n_agents: int = 2,
    heartbeat_path: Path | None = None,
    tmp_path: Path | None = None,
) -> tuple[SimulationKernel, list[Agent]]:
    if agents is None:
        agents = [_make_agent(f"agent_{i}") for i in range(n_agents)]
    world = _make_world()
    event_logger = EventLogger((tmp_path or Path("/tmp")) / "events.jsonl", overwrite=True)
    kernel = SimulationKernel(
        agents=agents,
        world=world,
        logger=event_logger,
        heartbeat_path=heartbeat_path,
    )
    return kernel, agents


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestWriteHeartbeat:
    def test_no_op_when_path_is_none(self):
        """_write_heartbeat silently does nothing when heartbeat_path is None."""
        kernel, _ = _make_kernel()
        assert kernel.heartbeat_path is None
        kernel._write_heartbeat()  # must not raise

    def test_creates_file_with_correct_fields(self, tmp_path):
        hb_path = tmp_path / "exp" / "heartbeat.json"
        kernel, _ = _make_kernel(heartbeat_path=hb_path, tmp_path=tmp_path)

        before = time.time()
        kernel._write_heartbeat()
        after = time.time()

        assert hb_path.exists()
        data = json.loads(hb_path.read_text())
        assert data["round_id"] == kernel.world.state.round_id
        assert data["n_agents"] == 2
        assert before <= data["ts"] <= after

    def test_creates_parent_dirs(self, tmp_path):
        hb_path = tmp_path / "a" / "b" / "c" / "heartbeat.json"
        kernel, _ = _make_kernel(heartbeat_path=hb_path, tmp_path=tmp_path)
        kernel._write_heartbeat()
        assert hb_path.exists()

    def test_swallows_io_errors_silently(self, tmp_path):
        """A heartbeat I/O failure must never crash the simulation."""
        hb_path = tmp_path / "heartbeat.json"
        kernel, _ = _make_kernel(heartbeat_path=hb_path, tmp_path=tmp_path)

        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            kernel._write_heartbeat()  # must not raise

    def test_overwrites_stale_heartbeat(self, tmp_path):
        hb_path = tmp_path / "heartbeat.json"
        kernel, _ = _make_kernel(heartbeat_path=hb_path, tmp_path=tmp_path)

        kernel._write_heartbeat()
        ts_first = json.loads(hb_path.read_text())["ts"]

        time.sleep(0.05)
        kernel._write_heartbeat()
        ts_second = json.loads(hb_path.read_text())["ts"]

        assert ts_second > ts_first


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_creates_json_file(self, tmp_path):
        kernel, _ = _make_kernel(tmp_path=tmp_path)
        ckpt = tmp_path / "checkpoint.json"
        kernel.save_checkpoint(ckpt)
        assert ckpt.exists()

    def test_round_id_matches_world_state(self, tmp_path):
        kernel, _ = _make_kernel(tmp_path=tmp_path)
        kernel.world.state.round_id = 7
        ckpt = tmp_path / "ckpt.json"
        kernel.save_checkpoint(ckpt)
        data = json.loads(ckpt.read_text())
        assert data["round_id"] == 7

    def test_all_agents_serialized(self, tmp_path):
        agents = [_make_agent(f"a{i}", wealth=float(10 * i)) for i in range(3)]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        kernel.save_checkpoint(ckpt)
        data = json.loads(ckpt.read_text())
        assert set(data["agents"].keys()) == {"a0", "a1", "a2"}

    def test_wealth_values_persisted(self, tmp_path):
        agents = [_make_agent("rich", wealth=999.0), _make_agent("poor", wealth=1.0)]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        kernel.save_checkpoint(ckpt)
        data = json.loads(ckpt.read_text())
        assert data["agents"]["rich"]["wealth"] == pytest.approx(999.0)
        assert data["agents"]["poor"]["wealth"] == pytest.approx(1.0)

    def test_creates_parent_dirs_automatically(self, tmp_path):
        kernel, _ = _make_kernel(tmp_path=tmp_path)
        ckpt = tmp_path / "deep" / "nested" / "ckpt.json"
        kernel.save_checkpoint(ckpt)
        assert ckpt.exists()

    def test_last_action_serialized(self, tmp_path):
        agents = [_make_agent("a0")]
        agents[0].state.last_action = "cooperate"
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        kernel.save_checkpoint(ckpt)
        data = json.loads(ckpt.read_text())
        assert data["agents"]["a0"]["last_action"] == "cooperate"

    def test_trust_dict_serialized(self, tmp_path):
        agents = [_make_agent("a0")]
        agents[0].state.trust = {"a1": 0.8, "a2": 0.3}
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        kernel.save_checkpoint(ckpt)
        data = json.loads(ckpt.read_text())
        assert data["agents"]["a0"]["trust"] == {"a1": pytest.approx(0.8), "a2": pytest.approx(0.3)}


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def _write_ckpt(self, path: Path, round_id: int, agents_data: dict) -> None:
        path.write_text(json.dumps({"round_id": round_id, "agents": agents_data}))

    def test_returns_saved_round_id(self, tmp_path):
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(ckpt, round_id=5, agents_data={"a0": {"wealth": 10.0}})
        returned = kernel.load_checkpoint(ckpt)
        assert returned == 5

    def test_restores_world_round_id(self, tmp_path):
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(ckpt, round_id=12, agents_data={})
        kernel.load_checkpoint(ckpt)
        assert kernel.world.state.round_id == 12

    def test_restores_agent_wealth(self, tmp_path):
        agents = [_make_agent("a0", wealth=10.0)]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(ckpt, round_id=3, agents_data={"a0": {"wealth": 777.0}})
        kernel.load_checkpoint(ckpt)
        assert agents[0].state.wealth == pytest.approx(777.0)

    def test_restores_stress_and_satisfaction(self, tmp_path):
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(
            ckpt,
            round_id=1,
            agents_data={"a0": {"wealth": 50.0, "stress": 0.7, "satisfaction": 0.3}},
        )
        kernel.load_checkpoint(ckpt)
        assert agents[0].state.stress == pytest.approx(0.7)
        assert agents[0].state.satisfaction == pytest.approx(0.3)

    def test_restores_last_action(self, tmp_path):
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(
            ckpt,
            round_id=2,
            agents_data={"a0": {"wealth": 50.0, "last_action": "steal"}},
        )
        kernel.load_checkpoint(ckpt)
        assert agents[0].state.last_action == "steal"

    def test_restores_trust_dict(self, tmp_path):
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(
            ckpt,
            round_id=4,
            agents_data={"a0": {"wealth": 50.0, "trust": {"b1": 0.9}}},
        )
        kernel.load_checkpoint(ckpt)
        assert agents[0].state.trust == {"b1": pytest.approx(0.9)}

    def test_unknown_agent_ids_skipped(self, tmp_path):
        """Checkpoint with extra agent IDs not in the kernel must not crash."""
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        self._write_ckpt(
            ckpt,
            round_id=1,
            agents_data={
                "a0": {"wealth": 42.0},
                "ghost": {"wealth": 999.0},  # not in kernel
            },
        )
        kernel.load_checkpoint(ckpt)
        assert agents[0].state.wealth == pytest.approx(42.0)

    def test_missing_agent_keeps_current_state(self, tmp_path):
        """Agent absent from checkpoint retains its current state."""
        agents = [_make_agent("a0", wealth=88.0), _make_agent("a1", wealth=22.0)]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "ckpt.json"
        # only a0 in checkpoint
        self._write_ckpt(ckpt, round_id=3, agents_data={"a0": {"wealth": 100.0}})
        kernel.load_checkpoint(ckpt)
        assert agents[0].state.wealth == pytest.approx(100.0)
        assert agents[1].state.wealth == pytest.approx(22.0)  # unchanged

    def test_roundtrip_save_load(self, tmp_path):
        """save_checkpoint → load_checkpoint restores exact state."""
        agents = [_make_agent("x", wealth=321.0)]
        agents[0].state.stress = 0.45
        agents[0].state.last_action = "save"
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        kernel.world.state.round_id = 9
        ckpt = tmp_path / "rt.json"

        kernel.save_checkpoint(ckpt)

        # Mutate state after saving
        agents[0].state.wealth = 0.0
        agents[0].state.stress = 0.0
        kernel.world.state.round_id = 0

        kernel.load_checkpoint(ckpt)

        assert agents[0].state.wealth == pytest.approx(321.0)
        assert agents[0].state.stress == pytest.approx(0.45)
        assert agents[0].state.last_action == "save"
        assert kernel.world.state.round_id == 9


# ---------------------------------------------------------------------------
# run() — resume and round counting
# ---------------------------------------------------------------------------


class TestCheckpointMemory:
    """FIX 1 (C1): agent HierarchicalMemory must survive checkpoint/resume."""

    def _populate(self, mem):
        from agents.memory import MemoryItem, MemoryLevel

        mem.add(MemoryItem(0, "a1", "cooperate", "helped a1", {"wealth_delta": -3}, 0.7, 15, 0))
        mem.add(MemoryItem(1, None, "work", "earned", {"wealth_delta": 10}, 0.5, 10, 1))
        mem.archive.append(MemoryItem(99, "a2", "save", "old event", {}, 0.4, None, 99))
        mem.reflections.append("Over 5 events you mostly worked.")
        mem._pending_buffer.append(MemoryItem(2, "a3", "cooperate", "pending", {}, 0.6, 12, 2))
        mem._current_round = 7
        mem.level = MemoryLevel.M2

    def test_memory_roundtrip_exact(self, tmp_path):
        agents = [_make_agent("m0")]
        self._populate(agents[0].memory)
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "mem.json"
        kernel.save_checkpoint(ckpt)

        # Wipe memory entirely, then restore.
        fresh = [_make_agent("m0")]
        kernel2, _ = _make_kernel(agents=fresh, tmp_path=tmp_path)
        kernel2.load_checkpoint(ckpt)

        m = fresh[0].memory
        assert [i.content for i in m.recent] == ["helped a1", "earned"]
        assert [i.event_type for i in m.recent] == ["cooperate", "work"]
        assert m.recent[0].outcome == {"wealth_delta": -3}
        assert m.recent[0].expires_at_round == 15
        assert [i.content for i in m.archive] == ["old event"]
        assert m.reflections == ["Over 5 events you mostly worked."]
        assert [i.content for i in m._pending_buffer] == ["pending"]
        assert m._current_round == 7
        assert int(m.level) == 2

    def test_no_amnesia_after_resume(self, tmp_path):
        """The original bug: resumed agents had empty memory."""
        agents = [_make_agent("m0")]
        self._populate(agents[0].memory)
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        ckpt = tmp_path / "mem.json"
        kernel.save_checkpoint(ckpt)

        fresh = [_make_agent("m0")]
        assert fresh[0].memory.recent == []  # starts amnesiac
        kernel2, _ = _make_kernel(agents=fresh, tmp_path=tmp_path)
        kernel2.load_checkpoint(ckpt)
        assert len(fresh[0].memory.recent) > 0  # memory restored


class TestCheckpointGraphCollectiveRNG:
    """FIX 2/3/4(C4): social graph, RNG states, collective memory persist."""

    def test_social_graph_roundtrip(self, tmp_path):
        import random as _random

        from agents.collective_memory import CollectiveMemory
        from bgf_logging.event_logger import EventLogger
        from environment.network import NetworkManager
        from simulation.kernel import SimulationKernel

        agents = [_make_agent(f"agent_{i}") for i in range(3)]
        world = _make_world()
        world.network_manager = NetworkManager.fully_connected(["agent_0", "agent_1", "agent_2"])
        world.network_manager.strengthen_edge("agent_0", "agent_1", increment=0.5)
        cm = CollectiveMemory()
        cm.record(1, "shock", "wealth shock hit", importance=0.9)

        logger = EventLogger(tmp_path / "e.jsonl", overwrite=True)
        kernel = SimulationKernel(agents=agents, world=world, logger=logger, collective_memory=cm)
        _random.seed(123)
        _random.random()  # advance RNG
        ckpt = tmp_path / "full.json"
        kernel.save_checkpoint(ckpt)

        expected_next = _random.random()

        # New world with empty graph + empty collective memory.
        agents2 = [_make_agent(f"agent_{i}") for i in range(3)]
        world2 = _make_world()
        world2.network_manager = NetworkManager.fully_connected([])
        cm2 = CollectiveMemory()
        kernel2 = SimulationKernel(agents=agents2, world=world2, logger=logger, collective_memory=cm2)
        kernel2.load_checkpoint(ckpt)

        assert world2.network_manager.get_edge_weight("agent_0", "agent_1") == pytest.approx(1.5)
        assert [f.content for f in cm2.snapshot()] == ["wealth shock hit"]
        # RNG restored → next draw matches the value captured at save time.
        assert _random.random() == pytest.approx(expected_next)

    def test_world_state_roundtrip(self, tmp_path):
        agents = [_make_agent("a0")]
        kernel, _ = _make_kernel(agents=agents, tmp_path=tmp_path)
        kernel.world.state.prices = {"food": 2.5}
        kernel.world.state.public_signal = {"economy": "crisis"}
        kernel.world.state.resources = {"jobs": 42.0}
        ckpt = tmp_path / "ws.json"
        kernel.save_checkpoint(ckpt)

        kernel.world.state.prices = {}
        kernel.world.state.public_signal = {}
        kernel.world.state.resources = {}
        kernel.load_checkpoint(ckpt)

        assert kernel.world.state.prices == {"food": 2.5}
        assert kernel.world.state.public_signal == {"economy": "crisis"}
        assert kernel.world.state.resources == {"jobs": 42.0}


class TestRunResume:
    def test_fresh_run_executes_all_rounds(self, tmp_path):
        kernel, agents = _make_kernel(tmp_path=tmp_path)
        kernel.run(num_rounds=3)
        assert len(kernel.round_metrics) == 3

    def test_start_round_skips_completed_rounds(self, tmp_path):
        """start_round=2 with num_rounds=5 should only run 3 rounds."""
        kernel, _ = _make_kernel(tmp_path=tmp_path)
        kernel.run(num_rounds=5, start_round=2)
        assert len(kernel.round_metrics) == 3

    def test_fully_complete_run_is_noop(self, tmp_path):
        """start_round == num_rounds → no rounds executed."""
        kernel, _ = _make_kernel(tmp_path=tmp_path)
        kernel.run(num_rounds=4, start_round=4)
        assert len(kernel.round_metrics) == 0

    def test_over_complete_start_round_is_noop(self, tmp_path):
        """start_round > num_rounds → no rounds executed."""
        kernel, _ = _make_kernel(tmp_path=tmp_path)
        kernel.run(num_rounds=3, start_round=10)
        assert len(kernel.round_metrics) == 0

    def test_heartbeat_written_per_round(self, tmp_path):
        hb_path = tmp_path / "heartbeat.json"
        kernel, _ = _make_kernel(heartbeat_path=hb_path, tmp_path=tmp_path)

        kernel.run(num_rounds=3)

        assert hb_path.exists()
        data = json.loads(hb_path.read_text())
        # After 3 rounds, heartbeat reflects final state
        assert data["n_agents"] == 2

    def test_checkpoint_written_per_round_when_heartbeat_path_set(self, tmp_path):
        hb_path = tmp_path / "heartbeat.json"
        kernel, _ = _make_kernel(heartbeat_path=hb_path, tmp_path=tmp_path)
        kernel.run(num_rounds=2)
        ckpt = tmp_path / "checkpoint.json"
        assert ckpt.exists()
        data = json.loads(ckpt.read_text())
        assert "agents" in data
        assert "round_id" in data

    def test_no_checkpoint_without_heartbeat_path(self, tmp_path):
        kernel, _ = _make_kernel(tmp_path=tmp_path)  # no heartbeat_path
        kernel.run(num_rounds=2)
        ckpt = tmp_path / "checkpoint.json"
        assert not ckpt.exists()

    def test_resume_after_save_continues_correctly(self, tmp_path):
        """Full resume scenario: run 2 rounds, checkpoint, resume for 3 more."""
        hb_path = tmp_path / "heartbeat.json"
        agents = [_make_agent(f"agent_{i}") for i in range(2)]

        kernel_a, agents = _make_kernel(agents=agents, heartbeat_path=hb_path, tmp_path=tmp_path)
        kernel_a.run(num_rounds=2)

        ckpt = tmp_path / "checkpoint.json"
        assert ckpt.exists()

        # New kernel picks up from checkpoint
        agents2 = [_make_agent(f"agent_{i}") for i in range(2)]
        kernel_b, _ = _make_kernel(agents=agents2, heartbeat_path=hb_path, tmp_path=tmp_path)
        saved_round = kernel_b.load_checkpoint(ckpt)
        assert saved_round == 2

        kernel_b.run(num_rounds=5, start_round=saved_round)
        # 5 - 2 = 3 more rounds
        assert len(kernel_b.round_metrics) == 3
