"""Tests for ablated prompt construction — each of the 6 ablation modes."""

import pytest

from agents.memory import HierarchicalMemory, MemoryItem
from agents.profile import AgentProfile
from agents.state import AgentState
from decision.ablated_llm_policy import AblatedLLMPolicy
from decision.system_prompts import BASE_SYSTEM_PROMPT, SYSTEM_PROMPT_NO_INSTITUTIONS


# ── Fixtures ─────────────────────────────────────────────────────────────────


class FakeBackend:
    """Minimal backend stub for testing prompt construction only."""
    def generate(self, messages, temperature=None):
        return '{"action_type": "work", "amount": 10}', 0.1


@pytest.fixture
def profile():
    return AgentProfile(
        agent_id="agent_42",
        age=30,
        gender=2,
        income=800.0,
        education="college",
        occupation="teacher",
        location="urban",
        trust_people=0.7,
        risk_tolerance=0.4,
        competitiveness=0.5,
        political_preference="center",
        social_class="middle",
    )


@pytest.fixture
def state():
    return AgentState(wealth=100.0, stress=0.3, satisfaction=0.6)


@pytest.fixture
def memory():
    mem = HierarchicalMemory(max_recent=5)
    mem.add(MemoryItem(round_id=1, event_type="work", partner_id=None, content="earned 10", outcome={}))
    mem.add(MemoryItem(round_id=2, event_type="cooperate", partner_id="agent_5", content="shared", outcome={}))
    return mem


@pytest.fixture
def context():
    return {
        "world": {"prices": {"food": 1.0}, "public_signal": {"economy": "stable"}},
        "network": {"neighbors": ["agent_5", "agent_7"]},
    }


@pytest.fixture
def backend():
    return FakeBackend()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build(ablation: str, backend, profile, state, memory, context):
    policy = AblatedLLMPolicy(backend=backend, ablation=ablation)
    return policy._build_ablated_prompt(profile, state, memory, context, round_id=3)


# ── Invalid ablation ────────────────────────────────────────────────────────


class TestAblationValidation:
    def test_rejects_invalid_ablation(self, backend):
        with pytest.raises(ValueError, match="Invalid ablation"):
            AblatedLLMPolicy(backend=backend, ablation="nonexistent")

    @pytest.mark.parametrize("mode", AblatedLLMPolicy.VALID_ABLATIONS)
    def test_all_valid_modes_accepted(self, backend, mode):
        policy = AblatedLLMPolicy(backend=backend, ablation=mode)
        assert policy.ablation == mode


# ── no_persona ───────────────────────────────────────────────────────────────


class TestNoPersona:
    def test_uses_anonymous_persona(self, backend, profile, state, memory, context):
        msgs = _build("no_persona", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        assert "anonymous participant" in user_content
        assert "agent_42" not in user_content.split("Round")[0]  # persona section only

    def test_includes_memory(self, backend, profile, state, memory, context):
        msgs = _build("no_persona", backend, profile, state, memory, context)
        assert "work" in msgs[1]["content"]  # memory entry

    def test_includes_neighbors(self, backend, profile, state, memory, context):
        msgs = _build("no_persona", backend, profile, state, memory, context)
        assert "agent_5" in msgs[1]["content"]

    def test_uses_base_system_prompt(self, backend, profile, state, memory, context):
        msgs = _build("no_persona", backend, profile, state, memory, context)
        assert msgs[0]["content"] == BASE_SYSTEM_PROMPT


# ── minimal_persona ──────────────────────────────────────────────────────────


class TestMinimalPersona:
    def test_has_age_and_gender(self, backend, profile, state, memory, context):
        msgs = _build("minimal_persona", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        assert "30" in user_content  # age
        assert "female" in user_content  # gender=2

    def test_no_detailed_attributes(self, backend, profile, state, memory, context):
        msgs = _build("minimal_persona", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        # Full persona would include trust level, education, etc. Minimal should not.
        assert "trust" not in user_content.split("Round 3.")[1].split("Current situation")[0].lower() or \
               "Your trust" not in user_content.split("Round 3.")[1].split("Current situation")[0]


# ── rich_persona ─────────────────────────────────────────────────────────────


class TestRichPersona:
    def test_includes_full_persona(self, backend, profile, state, memory, context):
        msgs = _build("rich_persona", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        assert "agent_42" in user_content
        assert "college" in user_content or "Education" in user_content

    def test_includes_all_blocks(self, backend, profile, state, memory, context):
        msgs = _build("rich_persona", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        assert "wealth=" in user_content  # state
        assert "agent_5" in user_content  # neighbors
        assert "work" in user_content  # memory


# ── no_memory ────────────────────────────────────────────────────────────────


class TestNoMemory:
    def test_omits_memory_block(self, backend, profile, state, memory, context):
        msgs = _build("no_memory", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        # Should not have "Your recent memories" or round references from memory
        assert "recent memories" not in user_content.lower()
        assert "Round 1:" not in user_content
        assert "Round 2:" not in user_content

    def test_has_full_persona(self, backend, profile, state, memory, context):
        msgs = _build("no_memory", backend, profile, state, memory, context)
        assert "agent_42" in msgs[1]["content"]

    def test_has_neighbors(self, backend, profile, state, memory, context):
        msgs = _build("no_memory", backend, profile, state, memory, context)
        assert "agent_5" in msgs[1]["content"]


# ── no_network ───────────────────────────────────────────────────────────────


class TestNoNetwork:
    def test_no_neighbor_ids(self, backend, profile, state, memory, context):
        msgs = _build("no_network", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        # The context block should say "You have no neighbors"
        assert "no neighbors" in user_content.lower()

    def test_has_full_persona(self, backend, profile, state, memory, context):
        msgs = _build("no_network", backend, profile, state, memory, context)
        assert "agent_42" in msgs[1]["content"]

    def test_has_memory(self, backend, profile, state, memory, context):
        msgs = _build("no_network", backend, profile, state, memory, context)
        # Memory should still be present
        user_content = msgs[1]["content"]
        assert "work" in user_content


# ── no_institutions ──────────────────────────────────────────────────────────


class TestNoInstitutions:
    def test_uses_no_institutions_system_prompt(self, backend, profile, state, memory, context):
        msgs = _build("no_institutions", backend, profile, state, memory, context)
        assert msgs[0]["content"] == SYSTEM_PROMPT_NO_INSTITUTIONS

    def test_has_full_persona(self, backend, profile, state, memory, context):
        msgs = _build("no_institutions", backend, profile, state, memory, context)
        assert "agent_42" in msgs[1]["content"]

    def test_has_memory_and_neighbors(self, backend, profile, state, memory, context):
        msgs = _build("no_institutions", backend, profile, state, memory, context)
        user_content = msgs[1]["content"]
        assert "agent_5" in user_content


# ── Cross-cutting checks ────────────────────────────────────────────────────


class TestCrossCutting:
    @pytest.mark.parametrize("mode", AblatedLLMPolicy.VALID_ABLATIONS)
    def test_all_modes_produce_two_messages(self, backend, profile, state, memory, context, mode):
        msgs = _build(mode, backend, profile, state, memory, context)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    @pytest.mark.parametrize("mode", AblatedLLMPolicy.VALID_ABLATIONS)
    def test_all_modes_end_with_json_instruction(self, backend, profile, state, memory, context, mode):
        msgs = _build(mode, backend, profile, state, memory, context)
        assert "Respond with ONLY the JSON" in msgs[1]["content"]

    @pytest.mark.parametrize("mode", AblatedLLMPolicy.VALID_ABLATIONS)
    def test_all_modes_include_round_id(self, backend, profile, state, memory, context, mode):
        msgs = _build(mode, backend, profile, state, memory, context)
        assert "Round 3." in msgs[1]["content"]

    @pytest.mark.parametrize("mode", AblatedLLMPolicy.VALID_ABLATIONS)
    def test_all_modes_include_state(self, backend, profile, state, memory, context, mode):
        msgs = _build(mode, backend, profile, state, memory, context)
        assert "wealth=" in msgs[1]["content"]
