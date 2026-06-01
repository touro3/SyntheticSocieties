from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from decision.schemas import ProposedAction
from environment.payoffs import DEFAULT_PAYOFFS, GamePayoffs

if TYPE_CHECKING:
    from agents.agent import Agent
    from environment.world_state import WorldState


class ValidationResult(BaseModel):
    valid: bool
    reason: str | None = None


class InstitutionManager:
    def __init__(self, payoffs: GamePayoffs | None = None) -> None:
        self.payoffs = payoffs or DEFAULT_PAYOFFS

    def validate(
        self,
        action: ProposedAction,
        agent: Agent,
        world_state: WorldState,
        agent_lookup: Mapping[str, Agent],
    ) -> ValidationResult:
        if action.amount is not None:
            try:
                action.amount = float(action.amount)
            except (TypeError, ValueError):
                return ValidationResult(valid=False, reason="non_numeric_amount")
            if action.amount < 0:
                return ValidationResult(valid=False, reason="negative_amount")

        # ── Adversarial hard-constraint ──────────────────────────────────
        # A bad-apple agent is physically incapable of any action other than
        # theft. This is enforced by the deterministic gate, NOT by prompting
        # — the LLM cannot escape it no matter what it proposes. Symmetrically,
        # steal is an adversarial-only action: a normal agent cannot steal.
        adversarial = getattr(agent, "is_adversarial", False) or getattr(agent.profile, "is_adversarial", False)
        if adversarial and action.action_type != "steal":
            return ValidationResult(valid=False, reason="adversarial_must_steal")
        if action.action_type == "steal" and not adversarial:
            return ValidationResult(valid=False, reason="steal_not_permitted")

        if action.action_type == "steal":
            if not action.target_agent_id:
                return ValidationResult(valid=False, reason="missing_target")
            if action.target_agent_id not in agent_lookup:
                return ValidationResult(valid=False, reason="unknown_target")
            if action.target_agent_id == agent.profile.agent_id:
                return ValidationResult(valid=False, reason="self_target_not_allowed")

        # work/save carry no amount gate by design: execute() assigns their
        # payoff deterministically from GamePayoffs and ignores action.amount,
        # so an inflated amount cannot affect the outcome. The amount gate is
        # only meaningful for value-transferring actions (cooperate/steal).

        if action.action_type == "cooperate":
            if not action.target_agent_id:
                return ValidationResult(valid=False, reason="missing_target")

            if action.target_agent_id not in agent_lookup:
                return ValidationResult(valid=False, reason="unknown_target")

            if action.target_agent_id == agent.profile.agent_id:
                return ValidationResult(valid=False, reason="self_target_not_allowed")

            if action.amount is None or action.amount <= 0:
                return ValidationResult(valid=False, reason="invalid_transfer_amount")

            if agent.state.wealth < action.amount:
                return ValidationResult(valid=False, reason="insufficient_wealth")

        if action.action_type == "communicate":
            if not action.target_agent_id:
                return ValidationResult(valid=False, reason="missing_target")

            if action.target_agent_id not in agent_lookup:
                return ValidationResult(valid=False, reason="unknown_target")

            if action.target_agent_id == agent.profile.agent_id:
                return ValidationResult(valid=False, reason="self_target_not_allowed")

        return ValidationResult(valid=True)

    def execute(
        self,
        action: ProposedAction,
        agent: Agent,
        world_state: WorldState,
        agent_lookup: Mapping[str, Agent],
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "agent_id": agent.profile.agent_id,
            "action_type": action.action_type,
            "target_agent_id": action.target_agent_id,
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "satisfaction_delta": 0.0,
            "target_wealth_delta": 0.0,
            "interaction_type": None,
            "round_id": world_state.round_id,
        }

        if action.action_type == "work":
            # Income is a fixed payoff assigned by the institution, NOT the
            # amount the agent proposed. The LLM cannot self-assign wealth by
            # emitting a large `amount` — work always pays exactly work_income.
            event["wealth_delta"] = self.payoffs.work_income
            event["stress_delta"] = self.payoffs.work_stress_increase
            # Earning provides mild satisfaction, reduced when stress is high
            event["satisfaction_delta"] = 0.05 if agent.state.stress < 0.5 else -0.05

        elif action.action_type == "save":
            event["wealth_delta"] = self.payoffs.save_wealth_delta
            event["stress_delta"] = self.payoffs.save_stress_relief
            # Resting and saving provides security-based satisfaction
            event["satisfaction_delta"] = 0.08

        elif action.action_type == "cooperate":
            amount = float(action.amount or 0.0)

            event["wealth_delta"] = -amount
            event["target_wealth_delta"] = amount * self.payoffs.cooperation_multiplier
            event["stress_delta"] = self.payoffs.cooperate_stress_relief
            # Social cooperation provides the strongest satisfaction boost
            event["satisfaction_delta"] = 0.12
            event["interaction_type"] = "cooperation"

        elif action.action_type == "communicate":
            # Communication has no economic effect — purely informational.
            event["interaction_type"] = "communication"
            event["message_summary"] = action.reasoning_summary

        elif action.action_type == "steal":
            # Zero-sum predatory transfer. The thief takes `amount` from the
            # target, bounded by what the target actually holds (no wealth is
            # created from nothing, and the target cannot go negative).
            requested = float(action.amount or 0.0)
            target = agent_lookup.get(action.target_agent_id) if action.target_agent_id is not None else None
            target_wealth = float(target.state.wealth) if target is not None else 0.0
            stolen = max(0.0, min(requested, target_wealth))
            event["wealth_delta"] = stolen
            event["target_wealth_delta"] = -stolen
            event["stress_delta"] = self.payoffs.steal_stress_increase
            event["satisfaction_delta"] = -0.05
            event["interaction_type"] = "theft"

        return event
