"""
Template behavior policy — ESS archetype-based decision making.

Defines 4 fixed behavioral templates derived from ESS persona clusters:
  - Cooperator: high-trust agents who prefer sharing
  - Worker: low-trust, industrious agents
  - Saver: risk-averse, conservative agents
  - Balanced: mixed-strategy agents

Unlike the data-driven policy (McFadden logit), template agents follow
deterministic threshold rules. This provides a stronger baseline than
random but weaker than LLM or data-driven policies.
"""

from __future__ import annotations

from decision.schemas import ProposedAction


class TemplatePolicy:
    """
    Policy that selects actions based on one of 4 behavioral templates.

    The template is chosen from the agent's ESS-derived persona attributes:
      - trust_people > 0.6 → Cooperator
      - risk_tolerance < 0.3 → Saver
      - competitiveness > 0.6 or trust_people < 0.3 → Worker
      - otherwise → Balanced
    """

    def propose_action(self, profile, state, memory, context, round_id: int) -> ProposedAction:
        template = self._classify_template(profile)
        neighbors = context.get("network", {}).get("neighbors", [])

        if template == "cooperator":
            return self._cooperator_action(state, neighbors)
        elif template == "saver":
            return self._saver_action(state)
        elif template == "worker":
            return self._worker_action(state)
        else:
            return self._balanced_action(state, neighbors, round_id)

    def _classify_template(self, profile) -> str:
        """Classify agent into a behavioral template from ESS attributes."""
        trust = getattr(profile, "trust_people", None) or 0.5
        risk = getattr(profile, "risk_tolerance", None) or 0.5
        comp = getattr(profile, "competitiveness", None) or 0.5

        if trust > 0.6:
            return "cooperator"
        if risk < 0.3:
            return "saver"
        if comp > 0.6 or trust < 0.3:
            return "worker"
        return "balanced"

    def _cooperator_action(self, state, neighbors: list[str]) -> ProposedAction:
        """Cooperator: prefers sharing, works only when poor."""
        if state.wealth < 30:
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="Template:cooperator — need resources before sharing.",
                confidence=0.8,
            )
        if neighbors and state.wealth >= 10:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=min(5.0, state.wealth * 0.1),
                reasoning_summary="Template:cooperator — sharing with neighbor.",
                confidence=0.9,
            )
        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary="Template:cooperator — no neighbors, saving instead.",
            confidence=0.6,
        )

    def _saver_action(self, state) -> ProposedAction:
        """Saver: risk-averse, prefers saving unless very poor."""
        if state.wealth < 20:
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="Template:saver — working to survive.",
                confidence=0.8,
            )
        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary="Template:saver — conserving resources.",
            confidence=0.9,
        )

    def _worker_action(self, state) -> ProposedAction:
        """Worker: industrious, always prefers working unless wealthy."""
        if state.wealth > 200:
            return ProposedAction(
                action_type="save",
                amount=5.0,
                reasoning_summary="Template:worker — wealthy enough, saving.",
                confidence=0.7,
            )
        return ProposedAction(
            action_type="work",
            amount=10.0,
            reasoning_summary="Template:worker — maximizing income.",
            confidence=0.9,
        )

    def _balanced_action(self, state, neighbors: list[str], round_id: int) -> ProposedAction:
        """Balanced: cycles through actions based on round modulo."""
        cycle = round_id % 3

        if cycle == 0:
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="Template:balanced — work cycle.",
                confidence=0.7,
            )
        elif cycle == 1 and neighbors and state.wealth >= 10:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[round_id % len(neighbors)],
                amount=5.0,
                reasoning_summary="Template:balanced — cooperation cycle.",
                confidence=0.7,
            )
        else:
            return ProposedAction(
                action_type="save",
                amount=5.0,
                reasoning_summary="Template:balanced — save cycle.",
                confidence=0.7,
            )
