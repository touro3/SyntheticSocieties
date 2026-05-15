"""Tests for the memory-deletion ablation cohort logic (Phase 3).

Guarantees the counterfactual is clean and reproducible:
  - the targeted betrayal is absent from the treatment cohort after wiping;
  - the control cohort retains it untouched;
  - the operation is deterministic and idempotent.
"""

from __future__ import annotations

import importlib

mod = importlib.import_module("scripts.run_memory_deletion_ablation")


def test_treatment_wiped_control_intact():
    control = mod._make_cohort(5, "ctrl")
    treatment = mod._make_cohort(5, "trt")

    removed = sum(mod.delete_betrayal_memories(a, partner_id=mod.BETRAYER) for a in treatment)
    assert removed == 5  # one seeded betrayal per agent

    for a in treatment:
        assert all(not (it.event_type == "cooperate" and it.partner_id == mod.BETRAYER) for it in a.memory.recent)
    for a in control:
        assert any(
            it.event_type == "cooperate" and it.partner_id == mod.BETRAYER and it.outcome.get("reciprocated") is False
            for it in a.memory.recent
        )


def test_deletion_idempotent_and_deterministic():
    treatment = mod._make_cohort(4, "trt")
    first = sum(mod.delete_betrayal_memories(a, partner_id=mod.BETRAYER) for a in treatment)
    second = sum(mod.delete_betrayal_memories(a, partner_id=mod.BETRAYER) for a in treatment)
    assert first == 4
    assert second == 0  # nothing left to remove → idempotent

    a1 = mod._make_cohort(3, "trt")
    a2 = mod._make_cohort(3, "trt")
    assert sum(mod.delete_betrayal_memories(a, mod.BETRAYER) for a in a1) == sum(
        mod.delete_betrayal_memories(a, mod.BETRAYER) for a in a2
    )
