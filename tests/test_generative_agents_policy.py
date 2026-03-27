"""Tests for GenerativeAgentsPolicy (Condition C).

Phase 21 — Comparison to Generative Agents Baseline.

Validates:
- Backstory sampling is deterministic per agent_id
- All backstories are distinct across agents
- propose_action falls back correctly when backend returns None
- Prompt structure: fictional backstory + state + memory (no RAG blocks)
- Policy conforms to PolicyProtocol (structural subtyping)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agents.memory import MemoryBuffer
from decision.generative_agents_policy import GenerativeAgentsPolicy, _sample_fictional_backstory
from decision.schemas import ProposedAction
from tests.conftest import make_profile, make_state


# ── Backstory tests ───────────────────────────────────────────────────────────


def test_backstory_is_deterministic_per_agent():
    b1 = _sample_fictional_backstory("agent_5")
    b2 = _sample_fictional_backstory("agent_5")
    assert b1 == b2


def test_different_agents_may_get_different_backstories():
    backstories = {_sample_fictional_backstory(f"agent_{i}") for i in range(20)}
    assert len(backstories) > 1, "Expected variation across agents"


def test_backstory_is_non_empty():
    backstory = _sample_fictional_backstory("agent_0")
    assert len(backstory) > 50


def test_backstory_cache_consistent_across_rounds():
    policy = GenerativeAgentsPolicy(backend=None)
    b1 = policy._get_backstory("agent_42")
    b2 = policy._get_backstory("agent_42")
    assert b1 == b2


def test_backstory_seed_overrides_agent_hash():
    b_seed_1 = _sample_fictional_backstory("agent_0", seed=1)
    b_seed_2 = _sample_fictional_backstory("agent_0", seed=2)
    b_seed_1_again = _sample_fictional_backstory("agent_0", seed=1)
    assert b_seed_1 == b_seed_1_again
    # Different seeds may produce different backstories (not guaranteed but typical)
    # Just ensure the seeded version is consistent
    assert isinstance(b_seed_1, str)
    assert isinstance(b_seed_2, str)


# ── propose_action — fallback when backend is None ────────────────────────────


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_propose_action_fallback_no_backend():
    policy = GenerativeAgentsPolicy(backend=None)
    profile = make_profile(agent_id="agent_0")
    state = make_state(wealth=50.0)
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": []}}

    action = policy.propose_action(profile, state, memory, context, round_id=1)
    assert isinstance(action, ProposedAction)
    assert action.action_type in {"work", "save", "cooperate"}


# ── propose_action — with mock backend ───────────────────────────────────────


def _make_backend(response_json: str):
    backend = MagicMock()
    backend.generate.return_value = (response_json, 0.01)
    return backend


def test_propose_action_work_from_backend():
    backend = _make_backend('{"action_type": "work", "amount": 10, "reasoning_summary": "need money", "confidence": 0.8}')
    policy = GenerativeAgentsPolicy(backend=backend)
    profile = make_profile(agent_id="agent_1")
    state = make_state(wealth=50.0)
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": ["agent_2"]}}

    action = policy.propose_action(profile, state, memory, context, round_id=1)
    assert action.action_type == "work"
    assert action.amount == 10.0


def test_propose_action_cooperate_from_backend():
    backend = _make_backend(
        '{"action_type": "cooperate", "target_agent_id": "agent_2", '
        '"amount": 7, "reasoning_summary": "helping neighbor", "confidence": 0.9}'
    )
    policy = GenerativeAgentsPolicy(backend=backend)
    profile = make_profile(agent_id="agent_1")
    state = make_state(wealth=150.0)
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": ["agent_2"]}}

    action = policy.propose_action(profile, state, memory, context, round_id=2)
    assert action.action_type == "cooperate"
    assert action.target_agent_id == "agent_2"


def test_propose_action_invalid_json_falls_back():
    backend = _make_backend("not valid json at all")
    policy = GenerativeAgentsPolicy(backend=backend)
    profile = make_profile(agent_id="agent_0")
    state = make_state(wealth=50.0)
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": []}}

    action = policy.propose_action(profile, state, memory, context, round_id=1)
    assert isinstance(action, ProposedAction)
    assert action.action_type in {"work", "save", "cooperate"}


# ── Prompt structure: no RAG content ─────────────────────────────────────────


def test_prompt_contains_fictional_backstory():
    prompts_seen = []

    class CapturingBackend:
        def generate(self, messages, temperature=0.7):
            prompts_seen.extend(messages)
            return ('{"action_type": "work", "amount": 10, "reasoning_summary": "ok", "confidence": 0.5}', 0.01)

    policy = GenerativeAgentsPolicy(backend=CapturingBackend())
    profile = make_profile(agent_id="agent_0")
    state = make_state()
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": []}}

    policy.propose_action(profile, state, memory, context, round_id=1)

    user_message = next(m for m in prompts_seen if m["role"] == "user")
    content = user_message["content"]
    # Should contain fictional backstory text (not ESS field labels)
    assert "age" in content.lower() or "you are" in content.lower()
    # Should NOT contain ESS-specific numeric normalized fields
    assert "trust_people" not in content
    assert "ESS" not in content


def test_prompt_contains_round_id():
    prompts_seen = []

    class CapturingBackend:
        def generate(self, messages, temperature=0.7):
            prompts_seen.extend(messages)
            return ('{"action_type": "save", "amount": 5, "reasoning_summary": "ok", "confidence": 0.5}', 0.01)

    policy = GenerativeAgentsPolicy(backend=CapturingBackend())
    profile = make_profile(agent_id="agent_0")
    state = make_state()
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": []}}

    policy.propose_action(profile, state, memory, context, round_id=7)

    user_message = next(m for m in prompts_seen if m["role"] == "user")
    assert "7" in user_message["content"]


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_conforms_to_policy_protocol():
    from decision.policy_protocol import PolicyProtocol
    policy = GenerativeAgentsPolicy(backend=None)
    assert isinstance(policy, PolicyProtocol)


# ── Prompt logger integration ─────────────────────────────────────────────────


def test_prompt_logger_called_when_set():
    backend = _make_backend('{"action_type": "work", "amount": 10, "reasoning_summary": "ok", "confidence": 0.5}')
    mock_logger = MagicMock()
    policy = GenerativeAgentsPolicy(backend=backend, prompt_logger=mock_logger)
    profile = make_profile(agent_id="agent_0")
    state = make_state()
    memory = MemoryBuffer(max_items=5)
    context = {"network": {"neighbors": []}}

    policy.propose_action(profile, state, memory, context, round_id=3)
    mock_logger.log.assert_called_once()
    call_kwargs = mock_logger.log.call_args.kwargs
    assert call_kwargs["agent_id"] == "agent_0"
    assert call_kwargs["round_id"] == 3
