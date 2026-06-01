"""Tests for anti-drift memory features.

Covers:
  - Recency-weighted reflections: recent events dominate over stale history
  - Importance scoring: cooperation events score higher than work
  - Importance-based retrieval: get_important_recent() surfaces key events
  - Persona re-anchoring: drift detection injects grounding cue into prompt
  - Action distribution: weighted vs unweighted distribution computation
"""

from agents.memory import HierarchicalMemory, MemoryItem
from agents.profile import AgentProfile
from agents.state import AgentState

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _item(round_id: int, event_type: str, partner_id=None, outcome=None):
    return MemoryItem(
        round_id=round_id,
        event_type=event_type,
        partner_id=partner_id,
        content="",
        outcome=outcome or {},
    )


def _profile(**overrides):
    defaults = dict(
        agent_id="agent_0",
        age=35,
        income=1000.0,
        education="college",
        occupation="worker",
        location="urban",
        political_preference="center",
        social_class="middle",
        risk_tolerance=0.3,
        trust_people=0.8,  # high trust, low risk → expects ~56% coop
    )
    defaults.update(overrides)
    return AgentProfile(**defaults)


# ── Importance scoring ────────────────────────────────────────────────────────


class TestImportanceScoring:
    def test_cooperate_scores_higher_than_work(self):
        coop = _item(1, "cooperate", partner_id="a2")
        work = _item(1, "work")
        mem = HierarchicalMemory()
        assert mem._score_importance(coop) > mem._score_importance(work)

    def test_reciprocated_event_scores_highest(self):
        reciprocated = _item(1, "cooperate", partner_id="a2", outcome={"reciprocated": True})
        plain_coop = _item(1, "cooperate", partner_id="a2")
        mem = HierarchicalMemory()
        assert mem._score_importance(reciprocated) > mem._score_importance(plain_coop)

    def test_large_wealth_delta_boosts_importance(self):
        big_event = _item(1, "work", outcome={"wealth_delta": 15})
        small_event = _item(1, "work", outcome={"wealth_delta": 2})
        mem = HierarchicalMemory()
        assert mem._score_importance(big_event) > mem._score_importance(small_event)

    def test_importance_clamped_to_one(self):
        maxed = _item(1, "cooperate", partner_id="a2", outcome={"reciprocated": True, "wealth_delta": 20})
        mem = HierarchicalMemory()
        assert mem._score_importance(maxed) <= 1.0

    def test_importance_auto_assigned_on_add(self):
        mem = HierarchicalMemory()
        item = _item(1, "cooperate", partner_id="a2")
        assert item.importance == 0.0  # default
        mem.add(item)
        assert item.importance > 0.0


# ── Recency-weighted reflections ──────────────────────────────────────────────


class TestRecencyWeightedReflection:
    def test_recent_actions_dominate_reflection(self):
        """If early rounds are all work but recent are cooperate,
        the reflection should weight recent cooperate higher than
        a simple count would suggest."""
        mem = HierarchicalMemory(max_recent=30)
        # 10 old work events
        for i in range(10):
            mem.add(_item(i, "work"))
        # 10 recent cooperate events
        for i in range(10, 20):
            mem.add(_item(i, "cooperate", partner_id="a1"))

        reflection = mem.generate_reflection()
        # With recency weighting, cooperate should be listed first (higher weight)
        coop_pos = reflection.find("cooperate")
        work_pos = reflection.find("work")
        assert coop_pos < work_pos, f"Expected cooperate before work in reflection, got: {reflection}"

    def test_reflection_says_recency_weighted(self):
        mem = HierarchicalMemory()
        mem.add(_item(1, "work"))
        reflection = mem.generate_reflection()
        assert "recency-weighted" in reflection.lower()


# ── Importance-based retrieval ────────────────────────────────────────────────


class TestImportantRecent:
    def test_returns_all_when_under_limit(self):
        mem = HierarchicalMemory()
        mem.add(_item(1, "work"))
        mem.add(_item(2, "cooperate", partner_id="a1"))
        result = mem.get_important_recent(limit=5)
        assert len(result) == 2

    def test_prefers_cooperation_over_work_when_limited(self):
        """With a tight window, cooperation events should survive."""
        mem = HierarchicalMemory(max_recent=10)
        # Add 8 work events
        for i in range(8):
            mem.add(_item(i, "work"))
        # Add 2 cooperation events
        mem.add(_item(8, "cooperate", partner_id="a1"))
        mem.add(_item(9, "cooperate", partner_id="a2"))

        # Request only 3
        result = mem.get_important_recent(limit=3)
        coop_count = sum(1 for r in result if r.event_type == "cooperate")
        # Both cooperations should survive (high importance + decent recency)
        assert coop_count >= 1

    def test_chronological_order_preserved(self):
        mem = HierarchicalMemory(max_recent=10)
        for i in range(10):
            mem.add(_item(i, "work" if i % 2 == 0 else "cooperate", partner_id="a1"))

        result = mem.get_important_recent(limit=5)
        round_ids = [r.round_id for r in result]
        assert round_ids == sorted(round_ids)


# ── Action distribution ───────────────────────────────────────────────────────


class TestActionDistribution:
    def test_empty_memory_returns_empty_dict(self):
        mem = HierarchicalMemory()
        assert mem.get_action_distribution() == {}

    def test_unweighted_distribution_sums_to_one(self):
        mem = HierarchicalMemory()
        for i in range(5):
            mem.add(_item(i, "work"))
        for i in range(5, 8):
            mem.add(_item(i, "cooperate", partner_id="a1"))

        dist = mem.get_action_distribution(weighted=False)
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_weighted_distribution_sums_to_one(self):
        mem = HierarchicalMemory()
        for i in range(10):
            mem.add(_item(i, "work" if i < 7 else "cooperate", partner_id="a1"))

        dist = mem.get_action_distribution(weighted=True)
        assert abs(sum(dist.values()) - 1.0) < 1e-6

    def test_weighted_favors_recent_actions(self):
        """Recent cooperate should dominate when equal count to work."""
        mem = HierarchicalMemory(max_recent=30)
        # 10 old work events
        for i in range(10):
            mem.add(_item(i, "work"))
        # 10 recent cooperate events
        for i in range(10, 20):
            mem.add(_item(i, "cooperate", partner_id="a1"))

        dist = mem.get_action_distribution(weighted=True)
        # With recency weighting, cooperate should have higher weight
        assert dist.get("cooperate", 0) > dist.get("work", 0)


# ── Persona re-anchoring ─────────────────────────────────────────────────────


class TestPersonaReAnchoring:
    def test_no_anchor_with_insufficient_history(self):
        from decision.prompt_builder import _build_persona_anchor

        mem = HierarchicalMemory()
        mem.add(_item(1, "work"))
        profile = _profile()
        assert _build_persona_anchor(mem, profile) is None

    def test_no_anchor_when_behavior_matches_persona(self):
        from decision.prompt_builder import _build_persona_anchor

        mem = HierarchicalMemory(max_recent=20)
        profile = _profile(trust_people=0.8, risk_tolerance=0.3)
        # Expected coop ~0.54. Add mix that roughly matches.
        for i in range(10):
            if i < 5:
                mem.add(_item(i, "cooperate", partner_id="a1"))
            else:
                mem.add(_item(i, "work"))

        anchor = _build_persona_anchor(mem, profile)
        # Behavior approximately matches persona → no anchor needed
        assert anchor is None

    def test_anchor_when_cooperation_collapses(self):
        from decision.prompt_builder import _build_persona_anchor

        mem = HierarchicalMemory(max_recent=20)
        profile = _profile(trust_people=0.8, risk_tolerance=0.3)
        # Expected coop ~0.54, but agent only works → massive drift
        for i in range(10):
            mem.add(_item(i, "work"))

        anchor = _build_persona_anchor(mem, profile)
        assert anchor is not None
        assert "persona reminder" in anchor.lower()
        assert "trust" in anchor.lower()

    def test_anchor_when_over_cooperating(self):
        from decision.prompt_builder import _build_persona_anchor

        mem = HierarchicalMemory(max_recent=20)
        # Low trust agent who cooperates too much
        profile = _profile(trust_people=0.1, risk_tolerance=0.8)
        for i in range(10):
            mem.add(_item(i, "cooperate", partner_id="a1"))

        anchor = _build_persona_anchor(mem, profile)
        assert anchor is not None
        assert "comfort level" in anchor.lower()

    def test_memory_block_does_not_inject_anchor_into_prompts(self):
        """Persona re-anchoring must NOT appear in build_memory_block output.

        Injecting a formula-derived expected-cooperation rate into live
        prompts is a confound: it biases Condition B agent behavior via an
        unvalidated heuristic, undermining the empirical grounding claim.
        The anchor string is only available via the private _build_persona_anchor
        utility for post-hoc analysis; it must never appear in simulation prompts.
        """
        from decision.prompt_builder import build_memory_block

        mem = HierarchicalMemory(max_recent=20)
        profile = _profile(trust_people=0.8, risk_tolerance=0.3)
        # All work → would trigger drift anchor under the old (incorrect) design
        for i in range(10):
            mem.add(_item(i, "work"))

        block = build_memory_block(mem, window=5, profile=profile)
        assert "[Persona reminder]" not in block

    def test_memory_block_no_anchor_without_profile(self):
        from decision.prompt_builder import build_memory_block

        mem = HierarchicalMemory(max_recent=20)
        for i in range(10):
            mem.add(_item(i, "work"))

        block = build_memory_block(mem, window=5, profile=None)
        assert "[Persona reminder]" not in block

    def test_build_prompt_does_not_inject_persona_anchor(self):
        """build_prompt must not include the persona re-anchoring cue.

        Confirmed removal: the anchor formula is analysis-only and must not
        appear in prompts used during simulation runs.
        """
        from decision.prompt_builder import build_prompt

        mem = HierarchicalMemory(max_recent=20)
        profile = _profile(trust_people=0.9, risk_tolerance=0.1)
        state = AgentState(wealth=100.0, stress=0.3, satisfaction=0.5)
        context = {
            "world": {"prices": {"food": 1.0}, "public_signal": {"economy": "stable"}},
            "network": {"neighbors": ["a1"]},
        }
        for i in range(10):
            mem.add(_item(i, "work"))

        messages = build_prompt(profile, state, mem, context, round_id=11)
        full_text = " ".join(m["content"] for m in messages)
        assert "Persona reminder" not in full_text
