"""Rule-Based ESS Policy (Condition D) — Phase 28.2.

A *deterministic*, non-LLM policy that derives decisions directly from an
agent's ESS profile attributes.  This is the third baseline in the BGF
comparison table:

  Condition A — Pure LLM (no grounding)
  Condition B — ESS-Grounded LLM
  Condition D — Rule-Based ESS (this file; no LLM)

Purpose
-------
Condition D answers: "Does LLM reasoning add value, or is the ESS data alone
sufficient?"  If Condition B outperforms Condition D on behavioral realism,
LLM reasoning contributes beyond the grounding data.  If they are comparable,
the ESS grounding is doing all the work and a simpler deterministic agent would
suffice.

Decision Logic
--------------
Cooperation probability is derived from the agent's ESS profile:

    coop_prob = clip(0.2 + 0.5·trust_people·(1−risk_tolerance)
                     + 0.15·social_activity, 0.05, 0.90)

Determinism is ensured by hashing (agent_id, round_id) to produce a
reproducible pseudo-random value in [0, 1].  This makes the policy fully
reproducible across seeds without global RNG state.

Priority rules (applied before the cooperation probability check):
  1. Poverty escape: work if wealth < WORK_WEALTH_THRESHOLD
  2. Crisis mode:    work if stress > STRESS_CRITICAL
  3. Cooperation:    cooperate if hash_val < coop_prob AND neighbors exist
  4. Saving:         save if hash_val < coop_prob + save_buffer
  5. Default:        work
"""

from __future__ import annotations

import hashlib
import struct

from decision.constants import (
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_RULE_CONFIDENCE,
    DEFAULT_SAVE_AMOUNT,
    DEFAULT_WORK_AMOUNT,
    STRESS_CRITICAL,
    WORK_WEALTH_THRESHOLD,
)
from decision.prompt_builder import get_neighbors
from decision.schemas import ProposedAction


def _deterministic_uniform(agent_id: str, round_id: int) -> float:
    """Hash (agent_id, round_id) to a reproducible float in [0, 1).

    Uses SHA-256 truncated to 4 bytes, then interpreted as a uint32 and
    divided by 2^32.  Collision probability is negligible for realistic
    agent population sizes and simulation horizons.
    """
    key = f"{agent_id}:{round_id}".encode()
    digest = hashlib.sha256(key).digest()
    uint32 = struct.unpack(">I", digest[:4])[0]
    return uint32 / 4_294_967_296.0  # 2^32


def _cooperation_probability(profile) -> float:
    """Derive per-agent cooperation probability from ESS profile attributes.

    Formula:
        coop_prob = 0.2 + 0.5·trust·(1−risk) + 0.15·social_activity

    All attributes default to 0.5 (mid-range ESS value) when absent.
    The result is clipped to [0.05, 0.90] to avoid degenerate agents.
    """
    trust = float(profile.trust_people if profile.trust_people is not None else 0.5)
    risk = float(profile.risk_tolerance if profile.risk_tolerance is not None else 0.5)
    social = float(profile.social_activity if profile.social_activity is not None else 0.5)

    prob = 0.2 + 0.5 * trust * (1.0 - risk) + 0.15 * social
    return max(0.05, min(0.90, prob))


class RuleBasedESSPolicy:
    """Deterministic ESS-grounded baseline (Condition D).

    Implements the ``PolicyProtocol`` interface expected by the simulation
    kernel, but requires no LLM backend.

    The policy is:
      - **Deterministic**: same inputs → same output every time.
      - **Grounded**: uses ESS profile attributes directly.
      - **Fast**: no network calls, O(1) per decision.
      - **GPU-free**: runs on any machine; ideal for baseline sweeps.
    """

    def propose_action(
        self,
        profile,
        state,
        memory,
        context,
        round_id: int,
    ) -> ProposedAction:
        """Return a deterministic action derived from the ESS profile.

        Args:
            profile: AgentProfile with ESS attributes.
            state: AgentState with current wealth and stress.
            memory: AgentMemory (unused in this policy).
            context: World context dict (used to extract neighbors).
            round_id: Current simulation round (used for hashing).

        Returns:
            ProposedAction with action_type in {"work", "save", "cooperate"}.
        """
        neighbors = get_neighbors(context)

        # ── Priority 1: poverty escape ────────────────────────────────────
        if state.wealth < WORK_WEALTH_THRESHOLD:
            return ProposedAction(
                action_type="work",
                amount=DEFAULT_WORK_AMOUNT,
                reasoning_summary=("ESS-rule: low wealth triggers work priority."),
                confidence=DEFAULT_RULE_CONFIDENCE,
            )

        # ── Priority 2: crisis mode (high stress) ─────────────────────────
        if state.stress > STRESS_CRITICAL:
            return ProposedAction(
                action_type="work",
                amount=DEFAULT_WORK_AMOUNT,
                reasoning_summary=("ESS-rule: high stress triggers defensive work."),
                confidence=DEFAULT_RULE_CONFIDENCE,
            )

        # ── ESS-derived cooperation probability ───────────────────────────
        coop_prob = _cooperation_probability(profile)
        # Save probability covers ~40% of the non-cooperation probability mass
        save_prob = coop_prob + (1.0 - coop_prob) * 0.40

        # Deterministic draw — reproducible across seeds
        h = _deterministic_uniform(profile.agent_id, round_id)

        # ── Priority 3: cooperate ─────────────────────────────────────────
        if h < coop_prob and neighbors:
            # Fallback check injected here to prevent NoneType formatting crashes
            safe_trust = float(profile.trust_people if profile.trust_people is not None else 0.5)

            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=DEFAULT_COOPERATE_AMOUNT,
                reasoning_summary=(f"ESS-rule: trust={safe_trust:.2f}, coop_prob={coop_prob:.2f} → cooperate."),
                confidence=DEFAULT_RULE_CONFIDENCE,
            )

        # ── Priority 4: save ──────────────────────────────────────────────
        if h < save_prob:
            return ProposedAction(
                action_type="save",
                amount=DEFAULT_SAVE_AMOUNT,
                reasoning_summary=(f"ESS-rule: coop_prob={coop_prob:.2f}, h={h:.2f} → save."),
                confidence=DEFAULT_RULE_CONFIDENCE,
            )

        # ── Default: work ─────────────────────────────────────────────────
        return ProposedAction(
            action_type="work",
            amount=DEFAULT_WORK_AMOUNT,
            reasoning_summary="ESS-rule: default to income generation.",
            confidence=DEFAULT_RULE_CONFIDENCE,
        )
