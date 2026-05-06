import pytest

from agents.collective_memory import CollectiveMemory
from agents.memory import MemoryBuffer
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.prompt_builder import build_prompt


def test_collective_memory_returns_top_context_by_importance():
    memory = CollectiveMemory()
    memory.record(1, "narrative", "A quiet rumor spreads.", importance=0.4)
    memory.record(2, "shock", "A major shock occurred.", importance=0.9)

    context = memory.get_context(max_items=1)

    assert context == ["Round 2 shock: A major shock occurred."]


def test_collective_memory_decays_and_prunes_dead_facts():
    memory = CollectiveMemory(half_life_rounds=1.0, prune_below=0.2)
    memory.record(1, "milestone", "Cooperation peaked.", importance=0.5)

    memory.advance_round(3)

    assert memory.get_context() == []


def test_collective_memory_contribute_adds_agent_belief():
    memory = CollectiveMemory()
    memory.contribute("agent_7", round_id=5, fact_text="Cooperation feels risky today.")

    facts = memory.snapshot()
    assert len(facts) == 1
    assert facts[0].fact_type == "agent_belief"
    assert "agent_7" in facts[0].content
    assert facts[0].importance == pytest.approx(0.4)


def test_build_prompt_injects_collective_context_when_present():
    profile = AgentProfile(
        agent_id="agent_0",
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="urban",
        political_preference="center",
        risk_tolerance=0.5,
        social_class="middle",
    )
    state = AgentState(wealth=100.0)
    memory = MemoryBuffer(max_items=5)
    context = {
        "world": {"prices": {"food": 1.0}, "public_signal": {"economy": "stable"}, "resources": {"jobs": 10}},
        "network": {"neighbors": []},
    }

    messages = build_prompt(
        profile,
        state,
        memory,
        context,
        round_id=1,
        collective_context=["Round 1 shock: A major shock occurred."],
    )

    user_text = messages[1]["content"]
    assert "Community knowledge" in user_text
    assert "A major shock occurred" in user_text
