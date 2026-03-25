"""Tests for template policy and ablated LLM policy."""
from agents.profile import AgentProfile
from agents.state import AgentState
from agents.memory import MemoryBuffer
from decision.template_policy import TemplatePolicy


def _make_profile(**kwargs):
    defaults = dict(
        agent_id="agent_0", age=35, income=1000.0,
        education="college", occupation="worker", location="urban",
        political_preference="center", social_class="middle",
        trust_people=0.5, risk_tolerance=0.5, competitiveness=0.5,
    )
    defaults.update(kwargs)
    return AgentProfile(**defaults)


def _make_state(**kwargs):
    defaults = dict(wealth=100.0, stress=0.5, satisfaction=0.3)
    defaults.update(kwargs)
    return AgentState(**defaults)


def _ctx(neighbors=None):
    return {"network": {"neighbors": neighbors or []}, "world": {}}


# ── TemplatePolicy tests ──


def test_template_cooperator():
    policy = TemplatePolicy()
    profile = _make_profile(trust_people=0.8)
    state = _make_state(wealth=100.0)

    action = policy.propose_action(profile, state, MemoryBuffer(), _ctx(["agent_1"]), 1)
    assert action.action_type == "cooperate"
    assert action.target_agent_id == "agent_1"


def test_template_cooperator_poor_works():
    policy = TemplatePolicy()
    profile = _make_profile(trust_people=0.8)
    state = _make_state(wealth=10.0)

    action = policy.propose_action(profile, state, MemoryBuffer(), _ctx(["agent_1"]), 1)
    assert action.action_type == "work"


def test_template_saver():
    policy = TemplatePolicy()
    profile = _make_profile(risk_tolerance=0.2)
    state = _make_state(wealth=100.0)

    action = policy.propose_action(profile, state, MemoryBuffer(), _ctx(), 1)
    assert action.action_type == "save"


def test_template_worker():
    policy = TemplatePolicy()
    profile = _make_profile(trust_people=0.2, competitiveness=0.7)
    state = _make_state(wealth=100.0)

    action = policy.propose_action(profile, state, MemoryBuffer(), _ctx(), 1)
    assert action.action_type == "work"


def test_template_balanced_cycles():
    policy = TemplatePolicy()
    profile = _make_profile()
    state = _make_state(wealth=100.0)

    r0 = policy.propose_action(profile, state, MemoryBuffer(), _ctx(["a1"]), 0)
    assert r0.action_type == "work"

    r1 = policy.propose_action(profile, state, MemoryBuffer(), _ctx(["a1"]), 1)
    assert r1.action_type == "cooperate"

    r2 = policy.propose_action(profile, state, MemoryBuffer(), _ctx(["a1"]), 2)
    assert r2.action_type == "save"


def test_template_classify():
    policy = TemplatePolicy()
    assert policy._classify_template(_make_profile(trust_people=0.8)) == "cooperator"
    assert policy._classify_template(_make_profile(risk_tolerance=0.2)) == "saver"
    assert policy._classify_template(_make_profile(competitiveness=0.7, trust_people=0.2)) == "worker"
    assert policy._classify_template(_make_profile()) == "balanced"


def test_ablated_llm_invalid_ablation():
    from decision.ablated_llm_policy import AblatedLLMPolicy
    import pytest

    with pytest.raises(ValueError, match="Invalid ablation"):
        AblatedLLMPolicy(backend=None, ablation="invalid")


def test_ablated_llm_valid_ablations():
    from decision.ablated_llm_policy import AblatedLLMPolicy

    for mode in AblatedLLMPolicy.VALID_ABLATIONS:
        policy = AblatedLLMPolicy(backend=None, ablation=mode)
        assert policy.ablation == mode
