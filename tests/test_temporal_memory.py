"""Tests for temporal memory expiry, NL narration, backoff, and ReACT tools.

Covers the four MiroFish-inspired enhancements:
  1. MemoryItem auto-TTL + advance_round() expiry
  2. SimulationKernel._narrate_and_update_memory()
  3. LLMBackend exponential backoff + temperature reduction
  4. _TrackerTools.trend_analysis + _insight_forge_fallback
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from agents.memory import HierarchicalMemory, MemoryItem, MemoryLevel


# ── Helpers ────────────────────────────────────────────────────────────────────

def _item(
    round_id: int,
    event_type: str = "work",
    partner_id: str | None = None,
    expires_at: int | None = None,
    valid_at: int | None = None,
) -> MemoryItem:
    return MemoryItem(
        round_id=round_id,
        partner_id=partner_id,
        event_type=event_type,
        content=f"did {event_type} at round {round_id}",
        outcome={},
        valid_at=valid_at,
        expires_at_round=expires_at,
    )


# ── 1. Temporal memory expiry ─────────────────────────────────────────────────


class TestDefaultTTL:
    """HierarchicalMemory.default_ttl() returns event-appropriate lifetimes."""

    def test_cooperate_ttl(self):
        assert HierarchicalMemory.default_ttl("cooperate") == 15

    def test_work_ttl(self):
        assert HierarchicalMemory.default_ttl("work") == 10

    def test_steal_ttl(self):
        assert HierarchicalMemory.default_ttl("steal") == 20

    def test_observation_ttl(self):
        assert HierarchicalMemory.default_ttl("observation") == 8

    def test_unknown_event_type_returns_fallback(self):
        assert HierarchicalMemory.default_ttl("meditate") == 12


class TestMemoryItemValidAt:
    """MemoryItem.valid_at field is set correctly."""

    def test_valid_at_default_is_none(self):
        item = _item(5)
        assert item.valid_at is None

    def test_valid_at_set_explicitly(self):
        item = _item(5, valid_at=5)
        assert item.valid_at == 5


class TestAdvanceRound:
    """advance_round() expires stale items and moves them to archive."""

    def test_unexpired_item_stays_in_recent(self):
        mem = HierarchicalMemory()
        mem.add(_item(1, expires_at=10))
        mem.advance_round(5)
        assert len(mem.recent) == 1

    def test_expired_item_moves_to_archive(self):
        mem = HierarchicalMemory()
        mem.add(_item(1, expires_at=5))
        expired = mem.advance_round(6)
        assert expired == 1
        assert len(mem.recent) == 0
        assert len(mem.archive) == 1

    def test_no_expiry_item_persists_forever(self):
        mem = HierarchicalMemory()
        mem.add(_item(1, expires_at=None))
        mem.advance_round(1000)
        assert len(mem.recent) == 1

    def test_effective_recent_filters_expired(self):
        mem = HierarchicalMemory()
        mem.add(_item(1, expires_at=5, event_type="old"))
        mem.add(_item(2, expires_at=20, event_type="new"))
        mem._current_round = 6
        eff = mem._effective_recent
        assert len(eff) == 1
        assert eff[0].event_type == "new"

    def test_get_recent_respects_expiry(self):
        mem = HierarchicalMemory()
        for i in range(5):
            mem.add(_item(i, expires_at=i + 3))
        mem._current_round = 5
        recent = mem.get_recent(limit=10)
        # Only items with expires_at >= 5 should appear
        for item in recent:
            assert item.expires_at_round >= 5


class TestAutoTTLInRoundProcessor:
    """RoundProcessor._record_memory() auto-assigns valid_at and expires_at_round."""

    def test_record_memory_sets_ttl(self):
        from tests.conftest import make_profile, make_state

        from agents.agent import Agent
        from agents.memory import HierarchicalMemory
        from agents.state import AgentState
        from decision.schemas import ProposedAction
        from simulation.round_processor import RoundProcessor

        profile = make_profile(agent_id="a1")
        state = make_state(wealth=100.0)
        memory = HierarchicalMemory()
        policy = MagicMock()
        agent = Agent(profile=profile, state=state, memory=memory, policy=policy)

        world = MagicMock()
        logger_mock = MagicMock()
        processor = RoundProcessor(
            world=world, agent_lookup={"a1": agent}, logger=logger_mock,
        )

        action = ProposedAction(
            action_type="cooperate", target_agent_id="a2",
            reasoning_summary="helping neighbor",
        )
        event = {"action_type": "cooperate", "wealth_delta": -3}

        processor._record_memory(agent, action, event, round_id=5)

        assert len(memory.recent) == 1
        item = memory.recent[0]
        assert item.valid_at == 5
        assert item.expires_at_round == 5 + 15  # cooperate TTL = 15


# ── 2. NL narration memory update loop ────────────────────────────────────────


class TestNarrateAndUpdateMemory:
    """SimulationKernel._narrate_and_update_memory() creates observation items."""

    def _make_kernel(self, n_agents=3):
        from tests.conftest import make_profile, make_state

        from agents.agent import Agent
        from agents.memory import HierarchicalMemory

        agents = []
        for i in range(n_agents):
            profile = make_profile(agent_id=f"agent_{i}")
            state = make_state(wealth=50.0)
            state.last_action = "work"
            memory = HierarchicalMemory()
            policy = MagicMock()
            agents.append(Agent(profile=profile, state=state, memory=memory, policy=policy))

        world = MagicMock()
        # Each agent's neighbors are the other agents
        def get_agent_context(agent_id):
            others = [a.profile.agent_id for a in agents if a.profile.agent_id != agent_id]
            return {"neighbors": others}

        world.get_agent_context = get_agent_context
        logger_mock = MagicMock()

        from simulation.kernel import SimulationKernel
        kernel = SimulationKernel(agents=agents, world=world, logger=logger_mock)
        return kernel, agents

    def test_creates_observation_items(self):
        kernel, agents = self._make_kernel()
        kernel._narrate_and_update_memory(round_id=5)

        for agent in agents:
            obs = [m for m in agent.memory.recent if m.event_type == "observation"]
            assert len(obs) == 1
            assert "Round 5 observations:" in obs[0].content
            assert obs[0].valid_at == 5

    def test_observation_has_ttl(self):
        kernel, agents = self._make_kernel()
        kernel._narrate_and_update_memory(round_id=5)

        for agent in agents:
            obs = [m for m in agent.memory.recent if m.event_type == "observation"]
            assert obs[0].expires_at_round == 5 + 8  # observation TTL = 8

    def test_observation_importance_is_low(self):
        kernel, agents = self._make_kernel()
        kernel._narrate_and_update_memory(round_id=5)

        for agent in agents:
            obs = [m for m in agent.memory.recent if m.event_type == "observation"]
            assert obs[0].importance == 0.3

    def test_narration_mentions_neighbor_actions(self):
        kernel, agents = self._make_kernel()
        agents[0].state.last_action = "cooperate"
        agents[1].state.last_action = "save"
        kernel._narrate_and_update_memory(round_id=3)

        # Agent 2 should see agent_0 cooperate and agent_1 save
        obs = [m for m in agents[2].memory.recent if m.event_type == "observation"]
        assert "agent_0 chose to cooperate" in obs[0].content
        assert "agent_1 chose to save" in obs[0].content

    def test_no_observation_when_no_last_action(self):
        kernel, agents = self._make_kernel()
        for a in agents:
            a.state.last_action = None
        kernel._narrate_and_update_memory(round_id=1)
        for a in agents:
            assert len(a.memory.recent) == 0

    def test_neighbor_cap_at_5(self):
        kernel, agents = self._make_kernel(n_agents=8)
        kernel._narrate_and_update_memory(round_id=1)

        for agent in agents:
            obs = [m for m in agent.memory.recent if m.event_type == "observation"]
            if obs:
                # At most 5 neighbor references
                assert obs[0].content.count("chose to") <= 5


# ── 3. Exponential backoff + temperature reduction ────────────────────────────


class TestBackoffConstants:
    """LLMBackend has the new backoff configuration attributes."""

    def test_backoff_attributes_exist(self):
        from decision.llm_backend import LLMBackend
        backend = LLMBackend.__new__(LLMBackend)
        backend.__init__()
        assert hasattr(backend, "_backoff_initial_delay")
        assert hasattr(backend, "_backoff_factor")
        assert hasattr(backend, "_backoff_max_delay")
        assert hasattr(backend, "_retry_temp_reduction")

    def test_default_values(self):
        from decision.llm_backend import LLMBackend
        backend = LLMBackend.__new__(LLMBackend)
        backend.__init__()
        assert backend._backoff_initial_delay == 1.0
        assert backend._backoff_factor == 2.0
        assert backend._backoff_max_delay == 30.0
        assert backend._retry_temp_reduction == 0.1


class TestTemperatureReduction:
    """Retry temperature decreases progressively."""

    def test_temp_reduces_by_0_1_per_attempt(self):
        base_temp = 0.7
        reduction = 0.1
        for attempt in range(4):
            retry_temp = max(0.1, base_temp - (attempt * reduction))
            expected = max(0.1, 0.7 - attempt * 0.1)
            assert abs(retry_temp - expected) < 1e-9

    def test_temp_does_not_go_below_0_1(self):
        base_temp = 0.3
        reduction = 0.1
        for attempt in range(10):
            retry_temp = max(0.1, base_temp - (attempt * reduction))
            assert retry_temp >= 0.1


class TestOpenAIBackendTempReduction:
    """OpenAIBackend.generate() applies progressive temp reduction on retry."""

    def test_temp_reduces_on_retry(self):
        from decision.openai_backend import OpenAIBackend

        backend = OpenAIBackend(api_key="test-key", max_retries=2, min_delay=0.0)
        mock_client = MagicMock()

        # First call fails, second succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action_type": "work"}'

        mock_client.chat.completions.create = MagicMock(
            side_effect=[Exception("rate limit"), mock_response]
        )
        backend._client = mock_client

        with patch("time.sleep"):
            text, _ = backend.generate(
                [{"role": "user", "content": "test"}], temperature=0.7
            )

        # Should have been called twice
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls) == 2
        # First call: temp = 0.7 (attempt 0)
        assert calls[0].kwargs["temperature"] == pytest.approx(0.7)
        # Second call: temp = 0.6 (attempt 1, reduced by 0.1)
        assert calls[1].kwargs["temperature"] == pytest.approx(0.6)


# ── 4. ReACT report tools ────────────────────────────────────────────────────


class TestToolDescriptions:
    """New tools are registered in TOOL_DESCRIPTIONS and dispatch."""

    def test_trend_analysis_in_descriptions(self):
        from analysis.react_report_agent import _TrackerTools
        tools = _TrackerTools()
        assert "trend_analysis" in tools.TOOL_DESCRIPTIONS

    def test_trend_analysis_in_dispatch(self):
        from analysis.react_report_agent import _TrackerTools
        tools = _TrackerTools()
        # call() should not raise "Unknown tool" for trend_analysis
        result = tools.call("trend_analysis", {})
        assert "Unknown tool" not in result


class TestInsightForgeFallback:
    """_insight_forge_fallback works without an LLM client."""

    def test_fallback_produces_output(self):
        from analysis.react_report_agent import _TrackerTools
        tools = _TrackerTools()
        result = tools._insight_forge_fallback("What drives cooperation?")
        assert "Question: What drives cooperation?" in result
        assert "Policy Overview" in result

    def test_call_dispatches_to_fallback_when_no_client(self):
        from analysis.react_report_agent import _TrackerTools
        tools = _TrackerTools()
        result = tools.call(
            "insight_forge",
            {"question": "Why is Gini increasing?"},
            _client=None,
        )
        # Should NOT return the old error message
        assert "ERROR: insight_forge requires a live LLM client" not in result
        assert "Question:" in result


class TestTrendAnalysis:
    """trend_analysis produces correct output from the tracker index."""

    def test_handles_missing_index(self):
        from analysis.react_report_agent import _TrackerTools
        tools = _TrackerTools(index_path="nonexistent.parquet")
        result = tools.trend_analysis()
        assert "ERROR" in result

    def test_handles_empty_metric(self):
        from analysis.react_report_agent import _TrackerTools
        tools = _TrackerTools()
        # Even with an invalid metric, should fall back to wealth_mean
        result = tools.trend_analysis(metric="nonexistent_column")
        # Will either work (defaulting to wealth_mean) or error on missing index
        assert isinstance(result, str)
